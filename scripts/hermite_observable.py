import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

import cupy as cp


def make_noise(shape, method, dtype=cp.float32):
    reps, chains, dim = shape
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        z = cp.random.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
    if method == "antithetic":
        if chains % 2:
            z = cp.random.standard_normal((reps, chains // 2, dim), dtype=dtype)
            extra = cp.random.standard_normal((reps, 1, dim), dtype=dtype)
            return cp.concatenate([z, -z, extra], axis=1)
        z = cp.random.standard_normal((reps, chains // 2, dim), dtype=dtype)
        return cp.concatenate([z, -z], axis=1)
    raise ValueError(f"unknown method {method}")


def normalized_probe(dim, seed):
    rng = cp.random.RandomState(seed)
    a = rng.standard_normal(dim, dtype=cp.float32)
    return a / cp.linalg.norm(a)


def run_config(cfg):
    cp.random.seed(cfg["seed"])
    dtype = cp.float32
    reps = cfg["reps"]
    chains = cfg["chains"]
    dim = cfg["dim"]
    method = cfg["method"]
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    probe = normalized_probe(dim, cfg["design_seed"] + 1729).astype(dtype)

    start = time.time()
    z = make_noise((reps, chains, dim), method, dtype=dtype)
    linear = z @ probe
    quadratic = (cp.sum(z * z, axis=2) - dim) / math.sqrt(2.0 * dim)
    obs = alpha * linear + beta * quadratic
    obs_mean = obs.mean(axis=1)
    chain_obs = obs[:, 0]
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start

    empirical = float((cp.var(obs_mean) / cp.var(chain_obs)).get())
    if method == "iid":
        theory = 1.0 / chains
    elif method == "zero_sum":
        theory = (beta * beta / max(1, chains - 1)) / (alpha * alpha + beta * beta)
    elif method == "antithetic":
        if chains % 2:
            # One unmatched chain contributes the linear chaos; all pairs
            # contribute the even quadratic chaos.
            lin_factor = 1.0 / (chains * chains)
            quad_factor = (2 * (chains - 1) + 1) / (chains * chains)
        else:
            lin_factor = 0.0
            quad_factor = 2.0 / chains
        theory = (alpha * alpha * lin_factor + beta * beta * quad_factor) / (
            alpha * alpha + beta * beta
        )
    else:
        theory = float("nan")
    return {
        **cfg,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "elapsed_sec": elapsed,
        "chain_var": float(cp.var(chain_obs).get()),
        "ensemble_var": float(cp.var(obs_mean).get()),
        "empirical_ratio_to_chain": empirical,
        "theory_ratio_to_chain": theory,
        "abs_error": abs(empirical - theory),
    }


def shard_configs(shard):
    base_seed = 500000 + 10000 * shard
    dims = [64, 256, 1024, 4096]
    chains_by_shard = {
        0: [3, 5, 7],
        1: [8, 16],
        2: [31, 63],
    }
    if shard not in chains_by_shard:
        raise ValueError("shard must be 0, 1, or 2")
    ratios = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    methods = ["iid", "zero_sum", "antithetic"]
    for i, (dim, chains, ratio, method) in enumerate(
        itertools.product(dims, chains_by_shard[shard], ratios, methods)
    ):
        yield {
            "dim": dim,
            "chains": chains,
            "alpha": 1.0,
            "beta": ratio,
            "method": method,
            "reps": max(4096, min(16384, 1048576 // dim)),
            "seed": base_seed + i,
            "design_seed": base_seed + i // len(methods),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    configs = list(shard_configs(args.shard))
    if args.limit is not None:
        configs = configs[: args.limit]
    for cfg in configs:
        cfg["seed"] += args.seed_offset
    with args.out.open("a", encoding="utf-8") as f:
        for cfg in configs:
            result = run_config(cfg)
            f.write(json.dumps(result, sort_keys=True) + "\n")
            f.flush()
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
