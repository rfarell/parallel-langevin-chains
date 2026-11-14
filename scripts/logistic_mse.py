import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

import cupy as cp


METHODS = ["iid", "zero_sum", "antithetic", "common"]


def logistic_data(dim, n, seed):
    rng = cp.random.RandomState(seed)
    x = rng.standard_normal((n, dim), dtype=cp.float32) / math.sqrt(dim)
    w_true = rng.standard_normal(dim, dtype=cp.float32)
    logits = x @ w_true
    p = 1.0 / (1.0 + cp.exp(-logits))
    y = (rng.random_sample(n, dtype=cp.float32) < p).astype(cp.float32)
    return x, y


def logistic_grad(x_state, data_x, data_y, prior):
    flat = x_state.reshape((-1, x_state.shape[-1]))
    logits = flat @ data_x.T
    probs = 1.0 / (1.0 + cp.exp(-logits))
    grad = (probs - data_y[None, :]) @ data_x / data_x.shape[0]
    grad = grad + prior * flat
    return grad.reshape(x_state.shape)


def make_noise(shape, method, dtype=cp.float32):
    reps, chains, dim = shape
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        z = cp.random.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
    if method == "common":
        z = cp.random.standard_normal((reps, 1, dim), dtype=dtype)
        return cp.repeat(z, chains, axis=1)
    if method == "antithetic":
        if chains % 2:
            z = cp.random.standard_normal((reps, chains // 2, dim), dtype=dtype)
            extra = cp.random.standard_normal((reps, 1, dim), dtype=dtype)
            return cp.concatenate([z, -z, extra], axis=1)
        z = cp.random.standard_normal((reps, chains // 2, dim), dtype=dtype)
        return cp.concatenate([z, -z], axis=1)
    raise ValueError(f"unknown method {method}")


def reference_mean(design):
    cp.random.seed(design["seed"] + 123456)
    dtype = cp.float32
    dim = design["dim"]
    data_x, data_y = logistic_data(dim, design["n_data"], design["design_seed"] + 31415)
    x = cp.zeros((design["ref_reps"], design["ref_chains"], dim), dtype=dtype)
    eta = design["eta"]
    sqrt_2eta = math.sqrt(2.0 * eta)
    acc = cp.zeros(dim, dtype=dtype)
    windows = 0
    for step in range(design["ref_steps"]):
        grad = logistic_grad(x, data_x, data_y, design["prior"])
        z = cp.random.standard_normal(x.shape, dtype=dtype)
        x = x - eta * grad + sqrt_2eta * z
        if step >= design["ref_burnin"]:
            acc += x.mean(axis=(0, 1))
            windows += 1
    return acc / windows, data_x, data_y


def run_method(design, ref_mean, data_x, data_y, method):
    cp.random.seed(design["seed"] + METHODS.index(method))
    dtype = cp.float32
    dim = design["dim"]
    reps = design["reps"]
    chains = design["chains"]
    eta = design["eta"]
    sqrt_2eta = math.sqrt(2.0 * eta)
    x = cp.zeros((reps, chains, dim), dtype=dtype)
    ens_acc = cp.zeros((reps, dim), dtype=dtype)
    single_acc = cp.zeros((reps, dim), dtype=dtype)
    windows = 0
    start = time.time()
    for step in range(design["steps"]):
        grad = logistic_grad(x, data_x, data_y, design["prior"])
        z = make_noise(x.shape, method, dtype=dtype)
        x = x - eta * grad + sqrt_2eta * z
        if step >= design["burnin"]:
            ens_acc += x.mean(axis=1)
            single_acc += x[:, 0, :]
            windows += 1
    ens_est = ens_acc / windows
    single_est = single_acc / windows
    ens_mean = ens_est.mean(axis=0)
    single_mean = single_est.mean(axis=0)
    ens_centered = ens_est - ens_mean[None, :]
    single_centered = single_est - single_mean[None, :]
    ens_err = ens_est - ref_mean[None, :]
    single_err = single_est - ref_mean[None, :]
    cp.cuda.Stream.null.synchronize()
    return {
        **design,
        "method": method,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "elapsed_sec": time.time() - start,
        "windows": windows,
        "mse": float(cp.mean(cp.sum(ens_err * ens_err, axis=1)).get()),
        "bias_sq": float(cp.sum((ens_mean - ref_mean) ** 2).get()),
        "trace_var": float(cp.mean(cp.sum(ens_centered * ens_centered, axis=1)).get()),
        "single_mse": float(cp.mean(cp.sum(single_err * single_err, axis=1)).get()),
        "single_bias_sq": float(cp.sum((single_mean - ref_mean) ** 2).get()),
        "single_trace_var": float(cp.mean(cp.sum(single_centered * single_centered, axis=1)).get()),
        "mse_to_single": float(
            (cp.mean(cp.sum(ens_err * ens_err, axis=1)) / cp.mean(cp.sum(single_err * single_err, axis=1))).get()
        ),
        "marginal_second": float(cp.mean(cp.sum(x[:, 0, :] * x[:, 0, :], axis=1) / dim).get()),
        "ensemble_spread": float(cp.mean(cp.sum((x - x.mean(axis=1)[:, None, :]) ** 2, axis=2) / dim).get()),
    }


def shard_designs(shard):
    dims_by_shard = {0: [64], 1: [128], 2: [256]}
    if shard not in dims_by_shard:
        raise ValueError("shard must be 0, 1, or 2")
    base_seed = 800000 + 10000 * shard
    chains = [5, 8, 16]
    priors = [0.3, 1.0]
    n_data = [1024]
    idx = 0
    for dim, k, prior, n in itertools.product(dims_by_shard[shard], chains, priors, n_data):
        reps = max(128, min(512, 32768 // dim))
        eta = 0.02 / (prior + 0.25)
        yield {
            "target": "logistic_mse",
            "dim": dim,
            "chains": k,
            "prior": prior,
            "n_data": n,
            "steps": 900,
            "burnin": 450,
            "ref_steps": 1800,
            "ref_burnin": 900,
            "ref_reps": max(64, reps // 2),
            "ref_chains": 16,
            "eta": eta,
            "reps": reps,
            "seed": base_seed + idx,
            "design_seed": base_seed + idx,
        }
        idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    designs = list(shard_designs(args.shard))
    if args.limit is not None:
        designs = designs[: args.limit]
    with args.out.open("a", encoding="utf-8") as f:
        for design in designs:
            design["seed"] += args.seed_offset
            ref, data_x, data_y = reference_mean(design)
            for method in METHODS:
                result = run_method(design, ref, data_x, data_y, method)
                f.write(json.dumps(result, sort_keys=True) + "\n")
                f.flush()
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
