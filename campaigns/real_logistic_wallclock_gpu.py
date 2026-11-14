#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np


METHODS = ["iid", "zero_sum", "antithetic", "common"]
DATASET_DEFAULTS = {
    "wdbc": {"steps": 900, "burnin": 450, "ref_steps": 1800, "ref_burnin": 900, "reps": 256},
    "spambase": {"steps": 900, "burnin": 450, "ref_steps": 1800, "ref_burnin": 900, "reps": 192},
    "adult": {"steps": 700, "burnin": 350, "ref_steps": 1400, "ref_burnin": 700, "reps": 64},
}
DATASET_SEEDS = {"wdbc": 910000, "spambase": 920000, "adult": 930000}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def script_hash():
    data = Path(__file__).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def gpu_metadata():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    selected = visible.split(",")[0].strip() if visible else "0"
    meta = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": visible or "all",
        "gpu_name": "unknown",
        "gpu_uuid": "unknown",
        "driver_version": "unknown",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": str(cp.cuda.runtime.runtimeGetVersion()),
    }
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        meta["gpu_name"] = decode(props.get("name", "unknown"))
    except Exception:
        pass
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = [line.strip() for line in query.splitlines() if line.strip()]
        selected_row = None
        for row in rows:
            parts = [part.strip() for part in row.split(",", 3)]
            if len(parts) == 4 and parts[0] == selected:
                selected_row = parts
                break
        if selected_row is None and rows:
            selected_row = [part.strip() for part in rows[0].split(",", 3)]
        if selected_row and len(selected_row) == 4:
            _, name, uuid, driver = selected_row
            meta["gpu_name"] = name
            meta["gpu_uuid"] = uuid
            meta["driver_version"] = driver
    except Exception:
        pass
    return meta


def make_noise(shape, method, dtype=cp.float32):
    _, chains, _ = shape
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        z = cp.random.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
    if method == "common":
        z = cp.random.standard_normal((shape[0], 1, shape[-1]), dtype=dtype)
        return cp.repeat(z, chains, axis=1)
    if method == "antithetic":
        z = cp.random.standard_normal((shape[0], chains // 2, shape[-1]), dtype=dtype)
        if chains % 2:
            extra = cp.random.standard_normal((shape[0], 1, shape[-1]), dtype=dtype)
            return cp.concatenate([z, -z, extra], axis=1)
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


def default_reps(dataset, chains, override):
    if override is not None:
        return override
    base = DATASET_DEFAULTS[dataset]["reps"]
    return max(24, base // max(1, chains // 8))


def initial_state(design, reps, chains, dim):
    init_rng = cp.random.RandomState(design["seed"] + 60617 + 1009 * chains)
    return design["init_scale"] * init_rng.standard_normal((reps, chains, dim), dtype=cp.float32)


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


def run_method(design, data_x, data_y, test_x, test_y, ref, method, chunk_rows, shash, meta):
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1) + 1009 * design["chains"])
    reps = design["reps"]
    chains = design["chains"]
    dim = design["dim"]
    x = initial_state(design, reps, chains, dim)
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
        **meta,
        "target": "real_logistic_wallclock_ula",
        "method": method,
        "coupling": method,
        "git_hash": git_hash(),
        "script_hash": shash,
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
        "cost_trace_time": float(cp.mean(cp.sum(ens_centered * ens_centered, axis=1)).get()) * (elapsed / reps),
    }


def config_list(mode):
    if mode == "full":
        return [
            {"chains": 7, "method": "iid"},
            {"chains": 7, "method": "zero_sum"},
            {"chains": 7, "method": "antithetic"},
            {"chains": 8, "method": "iid"},
            {"chains": 8, "method": "zero_sum"},
            {"chains": 8, "method": "antithetic"},
            {"chains": 16, "method": "iid"},
            {"chains": 32, "method": "iid"},
            {"chains": 64, "method": "iid"},
            {"chains": 128, "method": "iid"},
        ]
    if mode == "k5":
        return [
            {"chains": 5, "method": "iid"},
            {"chains": 5, "method": "zero_sum"},
            {"chains": 5, "method": "antithetic"},
        ]
    raise ValueError(f"unknown mode {mode}")


def prior_designs(dataset, dim, args):
    defaults = DATASET_DEFAULTS[dataset]
    priors = args.priors or [0.1, 0.3]
    for idx, prior in enumerate(priors):
        eta = args.eta if args.eta is not None else min(0.03 / (prior + 0.25), 0.08)
        yield {
            "dataset": dataset,
            "dim": dim,
            "prior": float(prior),
            "eta": eta,
            "steps": args.steps or defaults["steps"],
            "burnin": args.burnin or defaults["burnin"],
            "ref_steps": args.ref_steps or defaults["ref_steps"],
            "ref_burnin": args.ref_burnin or defaults["ref_burnin"],
            "ref_reps": args.ref_reps,
            "ref_chains": args.ref_chains,
            "replicate": args.replicate,
            "init_scale": args.init_scale,
            "seed": DATASET_SEEDS[dataset] + 10000 * args.replicate + idx,
            "design_seed": DATASET_SEEDS[dataset] + idx,
            "mode": args.mode,
            "block": args.block,
            "order_seed": args.order_seed,
        }


def quick_adjust(args):
    args.priors = args.priors or [0.3]
    args.reps = args.reps or 16
    args.steps = args.steps or 40
    args.burnin = args.burnin or 20
    args.ref_steps = args.ref_steps or 80
    args.ref_burnin = args.ref_burnin or 40
    args.ref_reps = min(args.ref_reps, 8)
    args.ref_chains = min(args.ref_chains, 8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wdbc", "spambase", "adult"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--init-scale", type=float, required=True)
    parser.add_argument("--mode", choices=["full", "k5"], required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--priors", nargs="+", type=float, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--ref-steps", type=int, default=None)
    parser.add_argument("--ref-burnin", type=int, default=None)
    parser.add_argument("--ref-reps", type=int, default=16)
    parser.add_argument("--ref-chains", type=int, default=16)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--order-seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        quick_adjust(args)
    if args.order_seed is None:
        args.order_seed = 800000 + args.replicate
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data_x, data_y, test_x, test_y = load_dataset(args.data_dir, args.dataset)
    meta = gpu_metadata()
    shash = script_hash()
    rng = np.random.default_rng(args.order_seed + DATASET_SEEDS[args.dataset])
    configs = config_list(args.mode)
    indexed_configs = list(enumerate(configs))
    rng.shuffle(indexed_configs)
    with args.out.open("a", encoding="utf-8") as f:
        for prior_design in prior_designs(args.dataset, int(data_x.shape[1]), args):
            ref = reference_mean(prior_design, data_x, data_y, args.chunk_rows)
            for order_index, (config_index, config) in enumerate(indexed_configs):
                design = {
                    **prior_design,
                    **config,
                    "config_index": int(config_index),
                    "order_index": int(order_index),
                    "reps": default_reps(args.dataset, int(config["chains"]), args.reps),
                }
                result = run_method(
                    design,
                    data_x,
                    data_y,
                    test_x,
                    test_y,
                    ref,
                    config["method"],
                    args.chunk_rows,
                    shash,
                    meta,
                )
                f.write(json.dumps(result, sort_keys=True) + "\n")
                f.flush()
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
