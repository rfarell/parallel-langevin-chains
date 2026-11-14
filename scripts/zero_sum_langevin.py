import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

import cupy as cp


def make_noise(shape, method, dtype=cp.float32):
    """Return Gaussian noise with standard normal marginals for each chain."""
    reps, chains, dim = shape
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        if chains < 2:
            raise ValueError("zero_sum requires at least two chains")
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


def make_initial_state(reps, chains, dim, cfg, dtype=cp.float32):
    """Return an initial ensemble with controlled cross-chain covariance."""
    mode = cfg.get("init_mode", "common_zero")
    scale = cfg.get("init_scale", 1.0)
    shape = (reps, chains, dim)
    if mode == "common_zero":
        return cp.zeros(shape, dtype=dtype)
    if mode == "common_random":
        x0 = scale * cp.random.standard_normal((reps, 1, dim), dtype=dtype)
        return cp.repeat(x0, chains, axis=1)
    if mode == "iid_random":
        return scale * cp.random.standard_normal(shape, dtype=dtype)
    if mode == "zero_sum_random":
        if chains < 2:
            raise ValueError("zero_sum_random requires at least two chains")
        x0 = cp.random.standard_normal(shape, dtype=dtype)
        x0 = x0 - x0.mean(axis=1, keepdims=True)
        return scale * x0 * math.sqrt(chains / (chains - 1))
    if mode == "antithetic_random":
        if chains % 2:
            return make_initial_state(reps, chains, dim, {**cfg, "init_mode": "zero_sum_random"}, dtype=dtype)
        half = scale * cp.random.standard_normal((reps, chains // 2, dim), dtype=dtype)
        return cp.concatenate([half, -half], axis=1)
    raise ValueError(f"unknown init_mode {mode}")


def diagonal_spectrum(dim, condition, dtype=cp.float32):
    if condition == 1:
        return cp.ones(dim, dtype=dtype)
    grid = cp.linspace(0.0, 1.0, dim, dtype=dtype)
    return cp.exp(math.log(condition) * grid).astype(dtype)


def normalized_probe(dim, seed):
    rng = cp.random.RandomState(seed)
    a = rng.standard_normal(dim, dtype=cp.float32)
    return a / cp.linalg.norm(a)


def gaussian_grad(x, h, beta):
    return x * h


def quartic_grad(x, h, beta):
    # Strongly convex perturbation: U(x)=0.5*x'Hx + beta/(4d)*sum_i x_i^4.
    return x * h + (beta / x.shape[-1]) * x * x * x


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


def run_config(cfg):
    cp.random.seed(cfg["seed"])
    dtype = cp.float32
    reps = cfg["reps"]
    chains = cfg["chains"]
    dim = cfg["dim"]
    steps = cfg["steps"]
    avg_start = cfg.get("avg_start", steps)
    eta = cfg["eta"]
    method = cfg["method"]
    target = cfg["target"]
    beta = cfg.get("beta", 0.0)
    h = diagonal_spectrum(dim, cfg["condition"], dtype=dtype)
    design_seed = cfg.get("design_seed", cfg["seed"])
    probe = normalized_probe(dim, design_seed + 1729).astype(dtype)
    x = make_initial_state(reps, chains, dim, cfg, dtype=dtype)
    init_ens_mean = x.mean(axis=1)
    init_probe = init_ens_mean @ probe
    init_spread = cp.mean(cp.sum((x - init_ens_mean[:, None, :]) ** 2, axis=2) / dim)
    sqrt_2eta = math.sqrt(2.0 * eta)
    data_x = data_y = None
    if target == "logistic":
        data_x, data_y = logistic_data(dim, cfg["n_data"], design_seed + 31415)
    avg_ens = cp.zeros((reps, dim), dtype=dtype) if avg_start < steps else None
    avg_chain = cp.zeros((reps, dim), dtype=dtype) if avg_start < steps else None
    avg_windows = 0

    start = time.time()
    for step in range(steps):
        if target == "gaussian":
            grad = gaussian_grad(x, h, beta)
        elif target == "quartic":
            grad = quartic_grad(x, h, beta)
        elif target == "logistic":
            grad = logistic_grad(x, data_x, data_y, cfg["prior"])
        else:
            raise ValueError(f"unknown target {target}")
        z = make_noise(x.shape, method, dtype=dtype)
        x = x - eta * grad + sqrt_2eta * z
        if avg_start <= step:
            ens_step = x.mean(axis=1)
            avg_ens += ens_step
            avg_chain += x[:, 0, :]
            avg_windows += 1
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start

    chain0 = x[:, 0, :]
    ens_mean = x.mean(axis=1)
    lin_chain = chain0 @ probe
    lin_ens = ens_mean @ probe
    quad_chain = cp.sum(chain0 * chain0, axis=1) / dim
    quad_each = cp.sum(x * x, axis=2) / dim
    quad_ens = quad_each.mean(axis=1)
    marginal_second = cp.mean(cp.sum(chain0 * chain0, axis=1) / dim)
    spread = cp.mean(cp.sum((x - ens_mean[:, None, :]) ** 2, axis=2) / dim)
    result = {
        **cfg,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "elapsed_sec": elapsed,
        "lin_chain_var": float(cp.var(lin_chain).get()),
        "lin_ens_var": float(cp.var(lin_ens).get()),
        "lin_ens_mean": float(cp.mean(lin_ens).get()),
        "quad_chain_var": float(cp.var(quad_chain).get()),
        "quad_ens_var": float(cp.var(quad_ens).get()),
        "marginal_second": float(marginal_second.get()),
        "ensemble_spread": float(spread.get()),
        "init_ens_var": float(cp.var(init_probe).get()),
        "init_ensemble_spread": float(init_spread.get()),
    }
    if avg_windows:
        avg_ens = avg_ens / avg_windows
        avg_chain = avg_chain / avg_windows
        avg_lin_chain = avg_chain @ probe
        avg_lin_ens = avg_ens @ probe
        result.update(
            {
                "avg_start": avg_start,
                "avg_windows": avg_windows,
                "avg_lin_chain_var": float(cp.var(avg_lin_chain).get()),
                "avg_lin_ens_var": float(cp.var(avg_lin_ens).get()),
                "avg_empirical_ratio_to_chain": float(
                    (cp.var(avg_lin_ens) / cp.var(avg_lin_chain)).get()
                ),
            }
        )
    if target == "gaussian":
        # Exact for common deterministic initialization. Random starts add a
        # separately contractive term handled in the paper theorem.
        if cfg.get("init_mode", "common_zero") == "common_zero":
            if method == "iid":
                result["theory_ratio_to_chain"] = 1.0 / chains
            elif method in {"zero_sum", "antithetic"}:
                result["theory_ratio_to_chain"] = 0.0
            elif method == "common":
                result["theory_ratio_to_chain"] = 1.0
    result["empirical_ratio_to_chain"] = (
        result["lin_ens_var"] / result["lin_chain_var"]
        if result["lin_chain_var"] > 0
        else float("nan")
    )
    result["quad_empirical_ratio_to_chain"] = (
        result["quad_ens_var"] / result["quad_chain_var"]
        if result["quad_chain_var"] > 0
        else float("nan")
    )
    return result


def shard_configs(shard):
    base_seed = 100000 + 10000 * shard
    methods = ["iid", "zero_sum", "antithetic", "common"]
    if shard == 0:
        dims = [64, 256, 1024, 4096]
        chains = [2, 4, 8, 16]
        conds = [1, 10, 100]
        for i, (dim, k, cond, method) in enumerate(itertools.product(dims, chains, conds, methods)):
            yield {
                "target": "gaussian",
                "dim": dim,
                "chains": k,
                "condition": cond,
                "method": method,
                "steps": 500,
                "eta": 0.05 / cond,
                "reps": max(256, min(4096, 262144 // dim)),
                "seed": base_seed + i,
                "design_seed": base_seed + i // len(methods),
            }
    elif shard == 1:
        dims = [64, 256, 1024, 2048]
        chains = [4, 8, 16]
        conds = [1, 30]
        betas = [0.01, 0.05, 0.2, 1.0, 3.0]
        for i, (dim, k, cond, beta, method) in enumerate(itertools.product(dims, chains, conds, betas, methods)):
            yield {
                "target": "quartic",
                "dim": dim,
                "chains": k,
                "condition": cond,
                "beta": beta,
                "method": method,
                "steps": 700,
                "eta": 0.03 / cond,
                "reps": max(256, min(4096, 262144 // dim)),
                "seed": base_seed + i,
                "design_seed": base_seed + i // len(methods),
            }
    elif shard == 2:
        dims = [64, 128, 256, 512]
        chains = [4, 8, 16]
        priors = [0.1, 0.3, 1.0]
        n_data = [512, 2048]
        for i, (dim, k, prior, n, method) in enumerate(itertools.product(dims, chains, priors, n_data, methods)):
            yield {
                "target": "logistic",
                "dim": dim,
                "chains": k,
                "condition": 1,
                "prior": prior,
                "n_data": n,
                "method": method,
                "steps": 450,
                "eta": 0.02 / (prior + 0.25),
                "reps": max(64, min(1024, 65536 // dim)),
                "seed": base_seed + i,
                "design_seed": base_seed + i // len(methods),
            }
    elif shard in {3, 4, 5}:
        dims = [64, 256, 1024, 4096]
        chains = [4, 8, 16]
        conds = [1, 10, 100]
        init_modes = ["common_zero", "iid_random", "zero_sum_random", "common_random"]
        init_scales = [0.25, 1.0, 3.0]
        step_grid = {3: 50, 4: 250, 5: 1000}
        for i, (dim, k, cond, init_mode, init_scale, method) in enumerate(
            itertools.product(dims, chains, conds, init_modes, init_scales, methods)
        ):
            yield {
                "target": "gaussian",
                "dim": dim,
                "chains": k,
                "condition": cond,
                "method": method,
                "steps": step_grid[shard],
                "eta": 0.05 / cond,
                "reps": max(256, min(4096, 262144 // dim)),
                "seed": base_seed + i,
                "design_seed": base_seed + i // len(methods),
                "init_mode": init_mode,
                "init_scale": init_scale,
            }
    elif shard == 6:
        offset = 0
        dims = [64, 256, 1024]
        chains = [3, 5, 7, 9, 15]
        conds = [1, 30]
        betas = [0.05, 0.2, 1.0, 3.0]
        for i, (dim, k, cond, beta, method) in enumerate(
            itertools.product(dims, chains, conds, betas, methods)
        ):
            steps = 700
            yield {
                "target": "quartic",
                "dim": dim,
                "chains": k,
                "condition": cond,
                "beta": beta,
                "method": method,
                "steps": steps,
                "avg_start": steps // 2,
                "eta": 0.03 / cond,
                "reps": max(256, min(4096, 262144 // dim)),
                "seed": base_seed + offset + i,
                "design_seed": base_seed + offset + i // len(methods),
            }
        offset += len(dims) * len(chains) * len(conds) * len(betas) * len(methods)
        dims = [64, 128, 256]
        priors = [0.1, 0.3, 1.0]
        n_data = [512, 2048]
        for i, (dim, k, prior, n, method) in enumerate(
            itertools.product(dims, chains, priors, n_data, methods)
        ):
            steps = 450
            yield {
                "target": "logistic",
                "dim": dim,
                "chains": k,
                "condition": 1,
                "prior": prior,
                "n_data": n,
                "method": method,
                "steps": steps,
                "avg_start": steps // 2,
                "eta": 0.02 / (prior + 0.25),
                "reps": max(64, min(1024, 65536 // dim)),
                "seed": base_seed + offset + i,
                "design_seed": base_seed + offset + i // len(methods),
            }
    elif shard == 7:
        dims = [64, 128, 256, 512, 1024, 2048, 4096]
        betas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
        methods = ["zero_sum", "iid"]
        for i, (dim, beta, method) in enumerate(itertools.product(dims, betas, methods)):
            yield {
                "target": "quartic",
                "dim": dim,
                "chains": 8,
                "condition": 1,
                "beta": beta,
                "method": method,
                "steps": 700,
                "avg_start": 350,
                "eta": 0.03,
                "reps": max(512, min(8192, 524288 // dim)),
                "seed": base_seed + i,
                "design_seed": base_seed + i // len(methods),
            }
    elif shard == 8:
        dims = [256, 1024]
        chains = [3, 5, 9, 15]
        conds = [1, 100]
        for i, (dim, k, cond, method) in enumerate(itertools.product(dims, chains, conds, methods)):
            steps = 500
            yield {
                "target": "gaussian",
                "dim": dim,
                "chains": k,
                "condition": cond,
                "method": method,
                "steps": steps,
                "avg_start": steps // 2,
                "eta": 0.05 / cond,
                "reps": max(256, min(4096, 262144 // dim)),
                "seed": base_seed + i,
                "design_seed": base_seed + i // len(methods),
            }
    else:
        raise ValueError("shard must be 0, 1, 2, 3, 4, 5, 6, 7, or 8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--steps-multiplier", type=float, default=1.0)
    parser.add_argument("--reps-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    configs = list(shard_configs(args.shard))
    if args.limit is not None:
        configs = configs[: args.limit]
    for cfg in configs:
        cfg["seed"] += args.seed_offset
        cfg["steps"] = max(1, int(round(cfg["steps"] * args.steps_multiplier)))
        cfg["reps"] = max(1, int(round(cfg["reps"] * args.reps_multiplier)))
    with args.out.open("a", encoding="utf-8") as f:
        for cfg in configs:
            result = run_config(cfg)
            f.write(json.dumps(result, sort_keys=True) + "\n")
            f.flush()
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
