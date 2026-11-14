"""Odd-K zero-sum baseline comparison on real logistic targets.

Compares the exchangeable zero-sum coupling C_star against two odd-K
baselines with equal ensemble weights:
  - pair_plus_one: floor(K/2) sign-flip pairs + 1 unmatched iid chain
    (conventional; not zero-sum, residual unmatched-chain noise).
  - pairs_plus_triplet: block zero-sum design; (K-3)/2 sign-flip pairs
    + one 3-chain simplex block (the K=3 C_star), so noise sums to zero
    across chains exactly.
Conventions (loading, metrics, seeds, JSONL) follow real_logistic_gpu.py.
"""

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np


METHODS = ["iid", "zero_sum", "pair_plus_one", "pairs_plus_triplet"]
ZERO_SUM_METHODS = {"zero_sum", "pairs_plus_triplet"}
DATASET_DEFAULTS = {
    "wdbc": {"steps": 900, "burnin": 450, "ref_steps": 1800, "ref_burnin": 900, "reps": 256},
    "spambase": {"steps": 900, "burnin": 450, "ref_steps": 1800, "ref_burnin": 900, "reps": 192},
    "adult": {"steps": 700, "burnin": 350, "ref_steps": 1400, "ref_burnin": 700, "reps": 64},
}
SEED_BASE = {"wdbc": 1010000, "spambase": 1020000, "adult": 1030000}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_noise(shape, method, dtype=cp.float32):
    reps, chains, dim = shape
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        z = cp.random.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
    if method == "pair_plus_one":
        if chains % 2 == 0:
            raise ValueError("pair_plus_one requires odd K")
        z = cp.random.standard_normal((reps, chains // 2, dim), dtype=dtype)
        extra = cp.random.standard_normal((reps, 1, dim), dtype=dtype)
        return cp.concatenate([z, -z, extra], axis=1)
    if method == "pairs_plus_triplet":
        if chains % 2 == 0 or chains < 5:
            raise ValueError("pairs_plus_triplet requires odd K >= 5")
        npairs = (chains - 3) // 2
        z = cp.random.standard_normal((reps, npairs, dim), dtype=dtype)
        trip = cp.random.standard_normal((reps, 3, dim), dtype=dtype)
        trip = (trip - trip.mean(axis=1, keepdims=True)) * math.sqrt(3.0 / 2.0)
        return cp.concatenate([z, -z, trip], axis=1)
    raise ValueError(f"unknown method {method}")


def selftest():
    cp.random.seed(12345)
    reps, dim = 60000, 16
    ok = True
    for chains in (7, 5):
        for method in METHODS:
            noise = make_noise((reps, chains, dim), method)
            var_by_chain = cp.var(noise, axis=(0, 2))
            max_var_dev = float(cp.abs(var_by_chain - 1.0).max().get())
            var_ok = max_var_dev < 0.02
            ok = ok and var_ok
            chain_sum = noise.sum(axis=1)
            max_abs_sum = float(cp.abs(chain_sum).max().get())
            rms_sum = float(cp.sqrt(cp.mean(chain_sum * chain_sum)).get())
            line = (f"K={chains} {method:>19}: max|var-1|={max_var_dev:.4f} "
                    f"({'PASS' if var_ok else 'FAIL'})  sum rms={rms_sum:.4f} max|sum|={max_abs_sum:.2e}")
            if method in ZERO_SUM_METHODS:
                sum_ok = max_abs_sum < 1e-4
                ok = ok and sum_ok
                line += f"  zero-sum {'PASS' if sum_ok else 'FAIL'}"
            print(line)
        flat = make_noise((reps, chains, dim), "pairs_plus_triplet").transpose(1, 0, 2).reshape(chains, -1)
        corr = np.corrcoef(cp.asnumpy(flat))
        print(f"K={chains} pairs_plus_triplet chain correlation matrix:")
        for row in corr:
            print("   " + " ".join(f"{v:+.3f}" for v in row))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def sigmoid(z):
    return 1.0 / (1.0 + cp.exp(-cp.clip(z, -30.0, 30.0)))


def add_intercept(x_np):
    ones = np.ones((x_np.shape[0], 1), dtype=np.float32)
    return np.concatenate([ones, x_np.astype(np.float32)], axis=1)


def load_dataset(data_dir, name):
    data = np.load(data_dir / f"{name}.npz", allow_pickle=True)
    x_train = cp.asarray(add_intercept(data["X_train"]), dtype=cp.float32)
    x_test = cp.asarray(add_intercept(data["X_test"]), dtype=cp.float32)
    y_train = cp.asarray(data["y_train"], dtype=cp.float32)
    y_test = cp.asarray(data["y_test"], dtype=cp.float32)
    return x_train, y_train, x_test, y_test


def normalized_probe(dim, seed):
    rng = cp.random.RandomState(seed)
    probe = rng.standard_normal(dim, dtype=cp.float32)
    probe[0] = 0.0
    return probe / cp.linalg.norm(probe)


def logistic_grad(x_state, data_x, data_y, prior, chunk_rows):
    flat = x_state.reshape((-1, x_state.shape[-1]))
    out = cp.empty_like(flat)
    prior_mask = cp.ones(flat.shape[-1], dtype=cp.float32)
    prior_mask[0] = 0.0
    n = data_x.shape[0]
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        part = flat[start:stop]
        probs = sigmoid(part @ data_x.T)
        grad = (probs - data_y[None, :]) @ data_x / n
        grad = grad + prior * part * prior_mask[None, :]
        out[start:stop] = grad
    return out.reshape(x_state.shape)


def predictive_logloss(estimates, data_x, data_y, chunk_rows=2048):
    losses = []
    eps = 1e-6
    for start in range(0, estimates.shape[0], chunk_rows):
        stop = min(start + chunk_rows, estimates.shape[0])
        probs = sigmoid(estimates[start:stop] @ data_x.T)
        loss = -(data_y[None, :] * cp.log(probs + eps) + (1 - data_y[None, :]) * cp.log(1 - probs + eps))
        losses.append(cp.mean(loss, axis=1))
    return cp.concatenate(losses)


def reference_mean(design, data_x, data_y, chunk_rows):
    cp.random.seed(design["seed"] + 77717)
    x = cp.zeros((design["ref_reps"], design["ref_chains"], design["dim"]), dtype=cp.float32)
    acc = cp.zeros(design["dim"], dtype=cp.float32)
    sqrt_2eta = math.sqrt(2.0 * design["eta"])
    windows = 0
    for step in range(design["ref_steps"]):
        grad = logistic_grad(x, data_x, data_y, design["prior"], chunk_rows)
        x = x - design["eta"] * grad + sqrt_2eta * cp.random.standard_normal(x.shape, dtype=cp.float32)
        if step >= design["ref_burnin"]:
            acc += x.mean(axis=(0, 1))
            windows += 1
    return acc / windows


def run_method(design, data_x, data_y, test_x, test_y, ref, method, chunk_rows):
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1))
    reps = design["reps"]
    chains = design["chains"]
    dim = design["dim"]
    x = cp.zeros((reps, chains, dim), dtype=cp.float32)
    ens_acc = cp.zeros((reps, dim), dtype=cp.float32)
    single_acc = cp.zeros((reps, dim), dtype=cp.float32)
    probe = normalized_probe(dim, design["design_seed"] + 1729)
    sqrt_2eta = math.sqrt(2.0 * design["eta"])
    windows = 0
    warm = x[: min(reps, 2)].copy()
    warm_grad = logistic_grad(warm, data_x, data_y, design["prior"], chunk_rows)
    warm = warm - design["eta"] * warm_grad + sqrt_2eta * make_noise(warm.shape, method)
    _ = predictive_logloss(warm.mean(axis=1), test_x, test_y)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for step in range(design["steps"]):
        grad = logistic_grad(x, data_x, data_y, design["prior"], chunk_rows)
        x = x - design["eta"] * grad + sqrt_2eta * make_noise(x.shape, method)
        if step >= design["burnin"]:
            ens_acc += x.mean(axis=1)
            single_acc += x[:, 0, :]
            windows += 1
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    ens = ens_acc / windows
    single = single_acc / windows
    ens_mean = ens.mean(axis=0)
    single_mean = single.mean(axis=0)
    ens_err = ens - ref[None, :]
    single_err = single - ref[None, :]
    ens_centered = ens - ens_mean[None, :]
    single_centered = single - single_mean[None, :]
    ens_logloss = predictive_logloss(ens, test_x, test_y)
    single_logloss = predictive_logloss(single, test_x, test_y)
    chain0 = x[:, 0, :]
    ens_terminal = x.mean(axis=1)
    lin_ens = ens @ probe
    lin_single = single @ probe
    return {
        **design,
        "target": "oddk_zerosum",
        "dataset": design["dataset"],
        "method": method,
        "coupling": method,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "elapsed_sec_per_ensemble": elapsed / reps,
        "n_ensembles": reps,
        "windows": windows,
        "mse": float(cp.mean(cp.sum(ens_err * ens_err, axis=1)).get()),
        "bias_sq": float(cp.sum((ens_mean - ref) ** 2).get()),
        "trace_var": float(cp.mean(cp.sum(ens_centered * ens_centered, axis=1)).get()),
        "single_mse": float(cp.mean(cp.sum(single_err * single_err, axis=1)).get()),
        "single_bias_sq": float(cp.sum((single_mean - ref) ** 2).get()),
        "single_trace_var": float(cp.mean(cp.sum(single_centered * single_centered, axis=1)).get()),
        "mse_to_single": float((cp.mean(cp.sum(ens_err * ens_err, axis=1)) / cp.mean(cp.sum(single_err * single_err, axis=1))).get()),
        "lin_ens_var": float(cp.var(lin_ens).get()),
        "lin_chain_var": float(cp.var(lin_single).get()),
        "test_logloss": float(cp.mean(ens_logloss).get()),
        "single_test_logloss": float(cp.mean(single_logloss).get()),
        "test_logloss_var": float(cp.var(ens_logloss).get()),
        "marginal_second": float(cp.mean(cp.sum(chain0 * chain0, axis=1) / dim).get()),
        "terminal_spread": float(cp.mean(cp.sum((x - ens_terminal[:, None, :]) ** 2, axis=2) / dim).get()),
    }


def make_designs(dataset, dim, args):
    defaults = DATASET_DEFAULTS[dataset]
    chains = args.chains or [7, 5]
    priors = args.priors or [0.1, 0.3]
    seed_base = SEED_BASE[dataset]
    idx = 0
    for prior in priors:
        eta = args.eta if args.eta is not None else min(0.03 / (prior + 0.25), 0.08)
        for k in chains:
            reps = args.reps or max(24, defaults["reps"] // max(1, k // 8))
            yield {
                "dataset": dataset,
                "dim": dim,
                "chains": k,
                "prior": prior,
                "eta": eta,
                "steps": args.steps or defaults["steps"],
                "burnin": args.burnin or defaults["burnin"],
                "ref_steps": args.ref_steps or defaults["ref_steps"],
                "ref_burnin": args.ref_burnin or defaults["ref_burnin"],
                "ref_reps": args.ref_reps,
                "ref_chains": args.ref_chains,
                "reps": reps,
                "replicate": args.replicate,
                "seed": seed_base + 10000 * args.replicate + idx,
                "design_seed": seed_base + idx,
            }
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["wdbc", "spambase", "adult"],
                        default=["wdbc", "spambase", "adult"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--chains", nargs="+", type=int, default=None)
    parser.add_argument("--priors", nargs="+", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--ref-steps", type=int, default=None)
    parser.add_argument("--ref-burnin", type=int, default=None)
    parser.add_argument("--ref-reps", type=int, default=16)
    parser.add_argument("--ref-chains", type=int, default=16)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.out is None:
        parser.error("--out is required unless --selftest")
    if args.out.exists():
        parser.error(f"{args.out} already exists; refusing to append to a prior run")
    if args.quick:
        args.priors = args.priors or [0.3]
        args.reps = args.reps or 16
        args.steps = args.steps or 40
        args.burnin = args.burnin or 20
        args.ref_steps = args.ref_steps or 80
        args.ref_burnin = args.ref_burnin or 40
        args.ref_reps = min(args.ref_reps, 8)
        args.ref_chains = min(args.ref_chains, 8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as f:
        for dataset in args.datasets:
            data_x, data_y, test_x, test_y = load_dataset(args.data_dir, dataset)
            designs = list(make_designs(dataset, int(data_x.shape[1]), args))
            if args.limit is not None:
                designs = designs[: args.limit]
            for design in designs:
                ref = reference_mean(design, data_x, data_y, args.chunk_rows)
                for method in args.methods:
                    result = run_method(design, data_x, data_y, test_x, test_y, ref, method, args.chunk_rows)
                    f.write(json.dumps(result, sort_keys=True) + "\n")
                    f.flush()
                    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
