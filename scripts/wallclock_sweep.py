import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import cupy as cp

from zero_sum_langevin import diagonal_spectrum, logistic_data, logistic_grad, make_noise, normalized_probe, quartic_grad


METHODS = ["iid", "zero_sum", "antithetic", "common"]


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def gaussian_grad(x, h):
    return x * h


def run_config(cfg):
    cp.random.seed(cfg["seed"])
    dtype = cp.float32
    reps = cfg["reps"]
    chains = cfg["chains"]
    dim = cfg["dim"]
    x = cp.zeros((reps, chains, dim), dtype=dtype)
    h = diagonal_spectrum(dim, cfg["condition"], dtype=dtype)
    probe = normalized_probe(dim, cfg["design_seed"] + 1729).astype(dtype)
    data_x = data_y = None
    if cfg["target"] == "synthetic_logistic":
        data_x, data_y = logistic_data(dim, cfg["n_data"], cfg["design_seed"] + 31415)
    sqrt_2eta = math.sqrt(2.0 * cfg["eta"])
    warm = x[: min(reps, 2)].copy()
    if cfg["target"] == "gaussian":
        warm_grad = gaussian_grad(warm, h)
    elif cfg["target"] == "quartic":
        warm_grad = quartic_grad(warm, h, cfg["beta"])
    elif cfg["target"] == "synthetic_logistic":
        warm_grad = logistic_grad(warm, data_x, data_y, cfg["prior"])
    warm = warm - cfg["eta"] * warm_grad + sqrt_2eta * make_noise(warm.shape, cfg["method"], dtype=dtype)
    _ = warm.mean(axis=1) @ probe
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(cfg["steps"]):
        if cfg["target"] == "gaussian":
            grad = gaussian_grad(x, h)
        elif cfg["target"] == "quartic":
            grad = quartic_grad(x, h, cfg["beta"])
        elif cfg["target"] == "synthetic_logistic":
            grad = logistic_grad(x, data_x, data_y, cfg["prior"])
        else:
            raise ValueError(f"unknown target {cfg['target']}")
        x = x - cfg["eta"] * grad + sqrt_2eta * make_noise(x.shape, cfg["method"], dtype=dtype)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    chain0 = x[:, 0, :]
    ens = x.mean(axis=1)
    lin_chain = chain0 @ probe
    lin_ens = ens @ probe
    return {
        **cfg,
        "coupling": cfg["method"],
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "elapsed_sec_per_ensemble": elapsed / reps,
        "n_ensembles": reps,
        "lin_chain_var": float(cp.var(lin_chain).get()),
        "lin_ens_var": float(cp.var(lin_ens).get()),
        "variance_ratio_to_chain": float((cp.var(lin_ens) / cp.var(lin_chain)).get()) if float(cp.var(lin_chain).get()) > 0 else float("nan"),
        "marginal_second": float(cp.mean(cp.sum(chain0 * chain0, axis=1) / dim).get()),
        "terminal_spread": float(cp.mean(cp.sum((x - ens[:, None, :]) ** 2, axis=2) / dim).get()),
    }


def make_configs(args):
    ks = args.chains or ([4, 8] if args.quick else [4, 8, 16, 32, 64, 128, 256])
    methods = args.methods or METHODS
    base_seed = {"gaussian": 710000, "quartic": 720000, "synthetic_logistic": 730000}[args.target]
    idx = 0
    for k in ks:
        for method in methods:
            if args.target == "gaussian":
                dim, condition, eta, steps, reps = 1024, 100, 0.01, 200, 2048
                extra = {}
            elif args.target == "quartic":
                dim, condition, eta, steps, reps = 1024, 30, 0.001, 300, 1536
                extra = {"beta": args.beta}
            else:
                dim, condition, eta, steps, reps = 128, 1, 0.02 / 1.25, 300, 768
                extra = {"prior": 1.0, "n_data": 512}
            if args.quick:
                steps = min(steps, 40)
                reps = min(reps, 64)
                dim = min(dim, 128)
            reps = args.reps or max(16, reps // max(1, k // 16))
            yield {
                "target": args.target,
                "dim": dim,
                "chains": k,
                "condition": condition,
                "eta": eta,
                "steps": steps,
                "reps": reps,
                "method": method,
                "replicate": args.replicate,
                "seed": base_seed + 10000 * args.replicate + idx,
                "design_seed": base_seed + idx // max(1, len(methods)),
                **extra,
            }
            idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["gaussian", "quartic", "synthetic_logistic"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--chains", nargs="+", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    configs = list(make_configs(args))
    if args.limit is not None:
        configs = configs[: args.limit]
    with args.out.open("a", encoding="utf-8") as f:
        for cfg in configs:
            result = run_config(cfg)
            f.write(json.dumps(result, sort_keys=True) + "\n")
            f.flush()
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
