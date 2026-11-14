#!/usr/bin/env python3
import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np


METHODS = ["iid_iid", "iid_zero", "balanced_iid", "balanced_zero", "iid_antithetic"]
DATASET_DEFAULTS = {
    "wdbc": {"steps": 3600, "burnin": 1800, "reps": 256, "primary_batch": 8, "small_batch": 4},
    "spambase": {"steps": 3600, "burnin": 1800, "reps": 192, "primary_batch": 64, "small_batch": 32},
    "adult": {"steps": 2800, "burnin": 1400, "reps": 64, "primary_batch": 512, "small_batch": 256},
}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def make_noise(shape, coupling, dtype=cp.float32, rng=None):
    rng = rng or cp.random
    _, chains, _ = shape
    if coupling == "iid":
        return rng.standard_normal(shape, dtype=dtype)
    if coupling == "zero_sum":
        z = rng.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
    if coupling == "antithetic":
        z = rng.standard_normal((shape[0], chains // 2, shape[-1]), dtype=dtype)
        if chains % 2:
            extra = rng.standard_normal((shape[0], 1, shape[-1]), dtype=dtype)
            return cp.concatenate([z, -z, extra], axis=1)
        return cp.concatenate([z, -z], axis=1)
    raise ValueError(f"unknown coupling {coupling}")


def method_parts(method):
    if method == "iid_iid":
        return "iid", "iid"
    if method == "iid_zero":
        return "iid", "zero_sum"
    if method == "balanced_iid":
        return "balanced", "iid"
    if method == "balanced_zero":
        return "balanced", "zero_sum"
    if method == "iid_antithetic":
        return "iid", "antithetic"
    raise ValueError(f"unknown method {method}")


def full_logistic_grad(x_state, data_x, data_y, prior, chunk_rows):
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


def iid_minibatches(reps, chains, n_data, batch_size, replacement=False, rng=None):
    rng = rng or cp.random
    if replacement:
        return rng.randint(0, n_data, size=(reps, chains, batch_size), dtype=cp.int32)
    keys = rng.random_sample((reps * chains, n_data), dtype=cp.float32)
    idx = cp.argpartition(keys, batch_size - 1, axis=1)[:, :batch_size]
    return idx.reshape(reps, chains, batch_size)


def balanced_minibatches(reps, chains, n_data, batch_size, rng=None):
    rng = rng or cp.random
    if chains * batch_size > n_data:
        raise ValueError("balanced minibatches require chains * batch_size <= n_data")
    keys = rng.random_sample((reps, n_data), dtype=cp.float32)
    idx = cp.argpartition(keys, chains * batch_size - 1, axis=1)[:, : chains * batch_size]
    return idx.reshape(reps, chains, batch_size)


def minibatch_grad(
    x_state,
    data_x,
    data_y,
    prior,
    batch_policy,
    batch_size,
    iid_batch_replacement=False,
    batch_rng=None,
):
    reps, chains, dim = x_state.shape
    n_data = data_x.shape[0]
    if batch_policy == "iid":
        idx = iid_minibatches(
            reps,
            chains,
            n_data,
            batch_size,
            replacement=iid_batch_replacement,
            rng=batch_rng,
        )
    elif batch_policy == "balanced":
        idx = balanced_minibatches(reps, chains, n_data, batch_size, rng=batch_rng)
    else:
        raise ValueError(f"unknown batch policy {batch_policy}")
    xb = data_x[idx]
    yb = data_y[idx]
    logits = cp.sum(x_state[:, :, None, :] * xb, axis=3)
    probs = sigmoid(logits)
    grad = cp.mean((probs - yb)[:, :, :, None] * xb, axis=2)
    prior_mask = cp.ones(dim, dtype=cp.float32)
    prior_mask[0] = 0.0
    return grad + prior * x_state * prior_mask[None, None, :]


def initial_state(design, reps, chains, dim):
    if design["init_mode"] == "zero":
        return cp.zeros((reps, chains, dim), dtype=cp.float32)
    if design["init_mode"] == "iid_normal":
        init_rng = cp.random.RandomState(design["seed"] + 60617)
        x = design["init_scale"] * init_rng.standard_normal((reps, chains, dim), dtype=cp.float32)
        return x.astype(cp.float32)
    raise ValueError(f"unknown init mode {design['init_mode']}")


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
        grad = full_logistic_grad(x, data_x, data_y, design["prior"], chunk_rows)
        x = x - design["eta"] * grad + sqrt_2eta * cp.random.standard_normal(x.shape, dtype=cp.float32)
        if step >= design["ref_burnin"]:
            acc += x.mean(axis=(0, 1))
            windows += 1
    return acc / windows


def run_method(design, data_x, data_y, test_x, test_y, ref, method, chunk_rows):
    batch_policy, coupling = method_parts(method)
    reps = design["reps"]
    chains = design["chains"]
    dim = design["dim"]
    x = initial_state(design, reps, chains, dim)
    method_index = METHODS.index(method) + 1
    if design["common_batch_stream"]:
        batch_seed = design["seed"] + 90917
    else:
        batch_seed = design["seed"] + 101 * method_index + 13
    batch_rng = cp.random.RandomState(batch_seed)
    noise_rng = cp.random.RandomState(design["seed"] + 101 * method_index + 29)
    ens_acc = cp.zeros((reps, dim), dtype=cp.float32)
    single_acc = cp.zeros((reps, dim), dtype=cp.float32)
    probe = normalized_probe(dim, design["design_seed"] + 1729)
    sqrt_2eta = math.sqrt(2.0 * design["eta"])
    windows = 0
    warm = x[: min(reps, 2)].copy()
    warm_grad = minibatch_grad(
        warm,
        data_x,
        data_y,
        design["prior"],
        batch_policy,
        design["batch_size"],
        design["iid_batch_replacement"],
        batch_rng=batch_rng,
    )
    warm = warm - design["eta"] * warm_grad + sqrt_2eta * make_noise(warm.shape, coupling, rng=noise_rng)
    _ = predictive_logloss(warm.mean(axis=1), test_x, test_y)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for step in range(design["steps"]):
        grad = minibatch_grad(
            x,
            data_x,
            data_y,
            design["prior"],
            batch_policy,
            design["batch_size"],
            design["iid_batch_replacement"],
            batch_rng=batch_rng,
        )
        x = x - design["eta"] * grad + sqrt_2eta * make_noise(x.shape, coupling, rng=noise_rng)
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
        "target": "real_logistic_sgld",
        "dataset": design["dataset"],
        "method": method,
        "batch_policy": batch_policy,
        "coupling": coupling,
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
    priors = args.priors or [0.1, 0.3]
    seed_base = {"wdbc": 710000, "spambase": 720000, "adult": 730000}[dataset]
    idx = 0
    for prior in priors:
        full_eta = min(0.03 / (prior + 0.25), 0.08)
        eta = args.eta if args.eta is not None else 0.25 * full_eta
        batch_size = args.batch_size or defaults[f"{args.batch_regime}_batch"]
        yield {
            "dataset": dataset,
            "dim": dim,
            "chains": args.chains,
            "prior": prior,
            "eta": eta,
            "full_gradient_eta": full_eta,
            "steps": args.steps or defaults["steps"],
            "burnin": args.burnin or defaults["burnin"],
            "ref_steps": args.ref_steps or 2 * (args.steps or defaults["steps"]),
            "ref_burnin": args.ref_burnin or 2 * (args.burnin or defaults["burnin"]),
            "ref_reps": args.ref_reps,
            "ref_chains": args.ref_chains,
            "reps": args.reps or defaults["reps"],
            "replicate": args.replicate,
            "batch_size": batch_size,
            "batch_regime": args.batch_regime,
            "iid_batch_replacement": args.iid_batch_replacement,
            "common_batch_stream": args.common_batch_stream,
            "init_mode": args.init_mode,
            "init_scale": args.init_scale,
            "seed": seed_base + 10000 * args.replicate + idx,
            "design_seed": seed_base + idx,
        }
        idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wdbc", "spambase", "adult"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicate", type=int, default=30)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--priors", nargs="+", type=float, default=None)
    parser.add_argument("--batch-regime", choices=["primary", "small"], default="primary")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iid-batch-replacement", action="store_true")
    parser.add_argument("--common-batch-stream", action="store_true")
    parser.add_argument("--init-mode", choices=["zero", "iid_normal"], default="zero")
    parser.add_argument("--init-scale", type=float, default=0.0)
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
        args.priors = args.priors or [0.3]
        args.reps = args.reps or 16
        args.steps = args.steps or 40
        args.burnin = args.burnin or 20
        args.ref_steps = args.ref_steps or 80
        args.ref_burnin = args.ref_burnin or 40
        args.ref_reps = min(args.ref_reps, 8)
        args.ref_chains = min(args.ref_chains, 8)
        args.batch_size = args.batch_size or 4
    designs = list(make_designs(args.dataset, int(data_x.shape[1]), args))
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
