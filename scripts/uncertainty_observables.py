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
    "wdbc": {"steps": 900, "burnin": 450, "reps": 256},
    "spambase": {"steps": 900, "burnin": 450, "reps": 192},
    "adult": {"steps": 700, "burnin": 350, "reps": 64},
}
SEED_BASE = {"wdbc": 710000, "spambase": 720000, "adult": 730000}


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


def test_subset(test_x, dataset, n_sub):
    n_test = int(test_x.shape[0])
    n_sub = min(n_sub, n_test)
    rng = np.random.default_rng(SEED_BASE[dataset] + 4242)
    idx = np.sort(rng.choice(n_test, size=n_sub, replace=False))
    return test_x[cp.asarray(idx)], n_sub


def run_method(design, data_x, data_y, x_sub, method, chunk_rows, rep_chunk):
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1))
    reps = design["reps"]
    chains = design["chains"]
    dim = design["dim"]
    thin = design["thin"]
    kept_steps = list(range(design["burnin"], design["steps"], thin))
    n_kept = len(kept_steps)
    kept_set = set(kept_steps)
    x = cp.zeros((reps, chains, dim), dtype=cp.float32)
    buf = cp.empty((reps, n_kept, chains, dim), dtype=cp.float32)
    sqrt_2eta = math.sqrt(2.0 * design["eta"])
    warm = x[: min(reps, 2)].copy()
    warm_grad = logistic_grad(warm, data_x, data_y, design["prior"], chunk_rows)
    warm = warm - design["eta"] * warm_grad + sqrt_2eta * make_noise(warm.shape, method)
    _ = sigmoid(warm.reshape((-1, dim)) @ x_sub.T)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    stored = 0
    for step in range(design["steps"]):
        grad = logistic_grad(x, data_x, data_y, design["prior"], chunk_rows)
        x = x - design["eta"] * grad + sqrt_2eta * make_noise(x.shape, method)
        if step in kept_set:
            buf[:, stored] = x
            stored += 1
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    pooled = buf.reshape((reps, n_kept * chains, dim))
    n_pool = n_kept * chains
    # (a) pooled per-coordinate posterior mean / variance estimate per ensemble
    m = pooled.mean(axis=1, dtype=cp.float64)
    second = cp.square(pooled).mean(axis=1, dtype=cp.float64)
    vhat = second - m * m
    # (b) equal-tail 90% interval width per coordinate per ensemble
    ciw = cp.empty((reps, dim), dtype=cp.float64)
    for r0 in range(0, reps, rep_chunk):
        r1 = min(r0 + rep_chunk, reps)
        q = cp.percentile(pooled[r0:r1], (5.0, 95.0), axis=1)
        ciw[r0:r1] = (q[1] - q[0]).astype(cp.float64)
    # (c) posterior predictive probability per test point per ensemble
    n_sub = int(x_sub.shape[0])
    pred = cp.empty((reps, n_sub), dtype=cp.float64)
    for r0 in range(0, reps, rep_chunk):
        r1 = min(r0 + rep_chunk, reps)
        probs = sigmoid(pooled[r0:r1].reshape((-1, dim)) @ x_sub.T)
        pred[r0:r1] = probs.reshape((r1 - r0, n_pool, n_sub)).mean(axis=1, dtype=cp.float64)
    # across-ensemble variance metrics (population variance, matching parent cp.var)
    m_centered = m - m.mean(axis=0)
    chain0 = x[:, 0, :]
    ens_terminal = x.mean(axis=1)
    return {
        **design,
        "target": "uncertainty_obs",
        "method": method,
        "coupling": method,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "elapsed_sec_per_ensemble": elapsed / reps,
        "n_ensembles": reps,
        "windows": n_kept,
        "n_pooled_samples": n_pool,
        "n_test_sub": n_sub,
        "trace_var": float(cp.mean(cp.sum(m_centered * m_centered, axis=1)).get()),
        "vhat_mean_sum": float(cp.sum(cp.mean(vhat, axis=0)).get()),
        "vhat_var_sum": float(cp.sum(cp.var(vhat, axis=0)).get()),
        "ciw_mean_sum": float(cp.sum(cp.mean(ciw, axis=0)).get()),
        "ciw_var_sum": float(cp.sum(cp.var(ciw, axis=0)).get()),
        "pred_mean_avg": float(cp.mean(pred).get()),
        "pred_var_sum": float(cp.sum(cp.var(pred, axis=0)).get()),
        "marginal_second": float(cp.mean(cp.sum(chain0 * chain0, axis=1) / dim).get()),
        "terminal_spread": float(cp.mean(cp.sum((x - ens_terminal[:, None, :]) ** 2, axis=2) / dim).get()),
    }


def make_designs(dataset, dim, args):
    defaults = DATASET_DEFAULTS[dataset]
    chains = args.chains or [8]
    priors = args.priors or [0.1, 0.3]
    idx = 0
    for prior in priors:
        eta = args.eta if args.eta is not None else min(0.03 / (prior + 0.25), 0.08)
        for k in chains:
            reps = args.reps or max(24, defaults["reps"] // max(1, k // 8))
            steps = args.steps or defaults["steps"]
            burnin = args.burnin or defaults["burnin"]
            # memory guard: thin the stored trajectory by 2 until the buffer fits
            thin = 1
            while reps * k * math.ceil((steps - burnin) / thin) * dim * 4 > args.mem_gb * 1024**3:
                thin *= 2
            yield {
                "dataset": dataset,
                "dim": dim,
                "chains": k,
                "prior": prior,
                "eta": eta,
                "steps": steps,
                "burnin": burnin,
                "reps": reps,
                "thin": thin,
                "replicate": args.replicate,
                "seed": SEED_BASE[dataset] + 10000 * args.replicate + idx,
                "design_seed": SEED_BASE[dataset] + idx,
            }
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["wdbc", "spambase", "adult"],
                        default=["wdbc", "spambase", "adult"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["iid", "zero_sum", "antithetic"])
    parser.add_argument("--chains", nargs="+", type=int, default=None)
    parser.add_argument("--priors", nargs="+", type=float, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--rep-chunk", type=int, default=16)
    parser.add_argument("--n-test-sub", type=int, default=128)
    parser.add_argument("--mem-gb", type=float, default=6.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.priors = args.priors or [0.3]
        args.reps = args.reps or 16
        args.steps = args.steps or 40
        args.burnin = args.burnin or 20
    with args.out.open("a", encoding="utf-8") as f:
        for dataset in args.datasets:
            data_x, data_y, test_x, _ = load_dataset(args.data_dir, dataset)
            x_sub, _ = test_subset(test_x, dataset, args.n_test_sub)
            for design in make_designs(dataset, int(data_x.shape[1]), args):
                for method in args.methods:
                    result = run_method(design, data_x, data_y, x_sub, method, args.chunk_rows, args.rep_chunk)
                    f.write(json.dumps(result, sort_keys=True) + "\n")
                    f.flush()
                    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
