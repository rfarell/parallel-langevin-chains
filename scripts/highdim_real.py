import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np


METHODS = ["iid", "zero_sum", "antithetic", "common"]
DATASET_DEFAULTS = {
    "mnist79": {"steps": 700, "burnin": 350, "ref_steps": 1400, "ref_burnin": 700, "reps": 64},
    "gisette": {"steps": 700, "burnin": 350, "ref_steps": 1400, "ref_burnin": 700, "reps": 32},
}
SEED_BASE = {"mnist79": 640000, "gisette": 650000}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_noise(shape, method, dtype=cp.float32):
    reps, chains, _ = shape
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        z = cp.random.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
    if method == "common":
        z = cp.random.standard_normal((reps, 1, shape[-1]), dtype=dtype)
        return cp.repeat(z, chains, axis=1)
    if method == "antithetic":
        if chains % 2:
            z = cp.random.standard_normal((reps, chains // 2, shape[-1]), dtype=dtype)
            extra = cp.random.standard_normal((reps, 1, shape[-1]), dtype=dtype)
            return cp.concatenate([z, -z, extra], axis=1)
        z = cp.random.standard_normal((reps, chains // 2, shape[-1]), dtype=dtype)
        return cp.concatenate([z, -z], axis=1)
    raise ValueError(f"unknown method {method}")


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
        "target": "real_logistic",
        "dataset": design["dataset"],
        "method": method,
        "coupling": method,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "elapsed_sec_per_ensemble": elapsed / reps,
        "n_ensembles": reps,
        "windows": windows,
        "chunk_rows": chunk_rows,
        "mempool_bytes": int(cp.get_default_memory_pool().total_bytes()),
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
    chains = args.chains or [8]
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
    parser.add_argument("--dataset", choices=["mnist79", "gisette"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, required=True)
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
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data_x, data_y, test_x, test_y = load_dataset(args.data_dir, args.dataset)
    if args.quick:
        args.chains = args.chains or [8]
        args.priors = args.priors or [0.3]
        args.reps = args.reps or 16
        args.steps = args.steps or 40
        args.burnin = args.burnin or 20
        args.ref_steps = args.ref_steps or 80
        args.ref_burnin = args.ref_burnin or 40
        args.ref_reps = min(args.ref_reps, 8)
        args.ref_chains = min(args.ref_chains, 8)
    designs = list(make_designs(args.dataset, int(data_x.shape[1]), args))
    if args.limit is not None:
        designs = designs[: args.limit]
    with args.out.open("a", encoding="utf-8") as f:
        for design in designs:
            ref = reference_mean(design, data_x, data_y, args.chunk_rows)
            for method in args.methods:
                result = run_method(design, data_x, data_y, test_x, test_y, ref, method, args.chunk_rows)
                f.write(json.dumps(result, sort_keys=True) + "\n")
                f.flush()
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
