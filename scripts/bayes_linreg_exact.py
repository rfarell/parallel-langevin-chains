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


def load_dataset(data_dir, name):
    data = np.load(data_dir / f"{name}.npz", allow_pickle=True)
    return (
        cp.asarray(data["X_train"], dtype=cp.float64),
        cp.asarray(data["y_train"], dtype=cp.float64),
        cp.asarray(data["X_test"], dtype=cp.float64),
        cp.asarray(data["y_test"], dtype=cp.float64),
    )


def normalized_probe(dim, seed, dtype):
    rng = cp.random.RandomState(seed)
    probe = rng.standard_normal(dim).astype(dtype)
    return probe / cp.linalg.norm(probe)


def gram_stats(data_x, data_y, chunk_rows):
    # Chunked float64 X^T X and X^T y over data rows.
    n, d = data_x.shape
    xtx = cp.zeros((d, d), dtype=cp.float64)
    xty = cp.zeros(d, dtype=cp.float64)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        part = data_x[start:stop]
        xtx += part.T @ part
        xty += part.T @ data_y[start:stop]
    return xtx, xty


def run_method(design, mats, ref_mu, test_x, test_y, method):
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1))
    dtype = cp.float64 if design["dtype"] == "float64" else cp.float32
    a_mat = mats["a"].astype(dtype)  # X^T X / (n sigma2)
    b_vec = mats["b"].astype(dtype)  # X^T y / (n sigma2)
    reps, chains, dim = design["reps"], design["chains"], design["dim"]
    lam = design["lam"]
    eta = design["eta"]
    sqrt_2eta = dtype(math.sqrt(2.0 * eta))
    x = cp.zeros((reps, chains, dim), dtype=dtype)
    ens_acc = cp.zeros((reps, dim), dtype=dtype)
    single_acc = cp.zeros((reps, dim), dtype=dtype)
    probe = normalized_probe(dim, design["design_seed"] + 1729, dtype)
    windows = 0
    warm = x[: min(reps, 2)].copy()
    warm = warm - eta * (warm @ a_mat - b_vec + lam * warm) + sqrt_2eta * make_noise(warm.shape, method, dtype)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for step in range(design["steps"]):
        grad = x @ a_mat - b_vec + lam * x
        x = x - eta * grad + sqrt_2eta * make_noise(x.shape, method, dtype)
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
    ens_err = (ens.astype(cp.float64) - ref_mu[None, :])
    single_err = (single.astype(cp.float64) - ref_mu[None, :])
    ens_centered = ens - ens_mean[None, :]
    single_centered = single - single_mean[None, :]
    test_mse = cp.mean((ens.astype(cp.float64) @ test_x.T - test_y[None, :]) ** 2, axis=1)
    chain0 = x[:, 0, :]
    ens_terminal = x.mean(axis=1)
    lin_ens = ens @ probe
    lin_single = single @ probe
    return {
        **design,
        "target": "bayes_linreg_exact",
        "method": method,
        "coupling": method,
        "kernel": "ula",
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "elapsed_sec_per_ensemble": elapsed / reps,
        "n_ensembles": reps,
        "windows": windows,
        "mse": float(cp.mean(cp.sum(ens_err * ens_err, axis=1)).get()),
        "bias_sq": float(cp.sum((ens_mean.astype(cp.float64) - ref_mu) ** 2).get()),
        "trace_var": float(cp.mean(cp.sum(ens_centered.astype(cp.float64) ** 2, axis=1)).get()),
        "single_mse": float(cp.mean(cp.sum(single_err * single_err, axis=1)).get()),
        "single_bias_sq": float(cp.sum((single_mean.astype(cp.float64) - ref_mu) ** 2).get()),
        "single_trace_var": float(cp.mean(cp.sum(single_centered.astype(cp.float64) ** 2, axis=1)).get()),
        "mse_to_single": float((cp.mean(cp.sum(ens_err * ens_err, axis=1)) / cp.mean(cp.sum(single_err * single_err, axis=1))).get()),
        "lin_ens_var": float(cp.var(lin_ens.astype(cp.float64)).get()),
        "lin_chain_var": float(cp.var(lin_single.astype(cp.float64)).get()),
        "test_mse": float(cp.mean(test_mse).get()),
        "test_mse_var": float(cp.var(test_mse).get()),
        "marginal_second": float(cp.mean(cp.sum(chain0 * chain0, axis=1) / dim).get()),
        "terminal_spread": float(cp.mean(cp.sum((x - ens_terminal[:, None, :]) ** 2, axis=2) / dim).get()),
        "mu_norm_sq": float(cp.sum(ref_mu * ref_mu).get()),
    }


def make_designs(args, dim, mats):
    idx = 0
    seed_base = 810000
    for lam in args.lams:
        precision_max = float(mats["a_eigmax"] + lam)
        eta = args.eta if args.eta is not None else args.eta_scale / precision_max
        for k in args.chains:
            yield {
                "dataset": args.dataset,
                "dim": dim,
                "chains": k,
                "lam": lam,
                "prior": lam,
                "sigma2": mats["sigma2"],
                "precision_eigmax": precision_max,
                "eta": eta,
                "dtype": args.dtype,
                "steps": args.steps,
                "burnin": args.burnin,
                "reps": args.reps,
                "replicate": args.replicate,
                "seed": seed_base + 10000 * args.replicate + idx,
                "design_seed": seed_base + idx,
            }
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="superconduct")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--chains", nargs="+", type=int, default=[8])
    parser.add_argument("--lams", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--reps", type=int, default=256)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--burnin", type=int, default=350)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--eta-scale", type=float, default=0.5)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--jitter", type=float, default=1e-8)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.reps = min(args.reps, 16)
        args.steps = min(args.steps, 60)
        args.burnin = min(args.burnin, 30)
        args.lams = args.lams if args.lams != [0.1, 0.3] else [0.3]
    data_x, data_y, test_x, test_y = load_dataset(args.data_dir, args.dataset)
    n, dim = data_x.shape
    xtx, xty = gram_stats(data_x, data_y, args.chunk_rows)
    # sigma2 from ridge-with-jitter OLS residuals, computed once in float64.
    w_ols = cp.linalg.solve(xtx + args.jitter * cp.trace(xtx) / dim * cp.eye(dim, dtype=cp.float64), xty)
    sigma2 = float(cp.mean((data_y - data_x @ w_ols) ** 2).get())
    a_mat = xtx / (n * sigma2)
    b_vec = xty / (n * sigma2)
    a_eigmax = float(cp.linalg.eigvalsh(a_mat).max().get())
    mats = {"a": a_mat, "b": b_vec, "sigma2": sigma2, "a_eigmax": a_eigmax}
    with args.out.open("a", encoding="utf-8") as f:
        for design in make_designs(args, dim, mats):
            # Closed-form Gaussian posterior mean: (X^T X/(n s2) + lam I) mu = X^T y/(n s2).
            ref_mu = cp.linalg.solve(a_mat + design["lam"] * cp.eye(dim, dtype=cp.float64), b_vec)
            for method in args.methods:
                result = run_method(design, mats, ref_mu, test_x, test_y, method)
                f.write(json.dumps(result, sort_keys=True) + "\n")
                f.flush()
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
