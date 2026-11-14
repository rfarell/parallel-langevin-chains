import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import cupy as cp
import numpy as np


METHODS = ["iid", "zero_sum", "antithetic"]
DATASETS = ["wdbc", "spambase", "adult"]
DATASET_DEFAULTS = {
    "wdbc": {"steps": 900, "burnin": 450, "ref_steps": 1800, "ref_burnin": 900, "reps": 256},
    "spambase": {"steps": 900, "burnin": 450, "ref_steps": 1800, "ref_burnin": 900, "reps": 192},
    "adult": {"steps": 700, "burnin": 350, "ref_steps": 1400, "ref_burnin": 700, "reps": 64},
}
# Spec scan {1, 0.5, 0.25, 0.125} x default extended upward: at the parent ULA
# step size MALA accepts ~0.99 on all three datasets, so [0.4, 0.8] is only
# reachable by scaling eta up.
ETA_SCAN = [16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def default_eta(prior):
    return min(0.03 / (prior + 0.25), 0.08)


def np_potential_grad(theta, data_x, data_y, prior):
    # CPU mirror of the GPU potential for the finite-difference selftest.
    z = theta @ data_x.T
    u_data = np.mean(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - data_y[None, :] * z, axis=-1)
    mask = np.ones(theta.shape[-1])
    mask[0] = 0.0
    u = u_data + 0.5 * prior * np.sum((theta * mask[None, :]) ** 2, axis=-1)
    probs = 1.0 / (1.0 + np.exp(-z))
    grad = (probs - data_y[None, :]) @ data_x / data_x.shape[0] + prior * theta * mask[None, :]
    return u, grad


def selftest():
    rng = np.random.default_rng(0)
    n, d, prior = 32, 5, 0.3
    data_x = rng.standard_normal((n, d))
    data_x[:, 0] = 1.0
    data_y = (rng.random(n) < 0.5).astype(np.float64)
    theta = rng.standard_normal((3, d)) * 0.5
    u, grad = np_potential_grad(theta, data_x, data_y, prior)
    fd = np.zeros_like(grad)
    h = 1e-6
    for j in range(d):
        e = np.zeros(d)
        e[j] = h
        up, _ = np_potential_grad(theta + e[None, :], data_x, data_y, prior)
        um, _ = np_potential_grad(theta - e[None, :], data_x, data_y, prior)
        fd[:, j] = (up - um) / (2 * h)
    grad_err = float(np.max(np.abs(fd - grad)))
    assert grad_err < 1e-5, f"finite-difference gradient check failed: {grad_err}"
    # MH log-ratio: implemented quadratic form must match explicit proposal log-densities.
    eta = 0.05
    x = theta
    ux, gx = np_potential_grad(x, data_x, data_y, prior)
    y = x - eta * gx + math.sqrt(2 * eta) * rng.standard_normal(x.shape)
    uy, gy = np_potential_grad(y, data_x, data_y, prior)
    log_alpha = ux - uy + (np.sum((y - x + eta * gx) ** 2, axis=-1) - np.sum((x - y + eta * gy) ** 2, axis=-1)) / (4 * eta)
    log_q_fwd = -np.sum((y - (x - eta * gx)) ** 2, axis=-1) / (4 * eta)
    log_q_rev = -np.sum((x - (y - eta * gy)) ** 2, axis=-1) / (4 * eta)
    ref = (ux - uy) + (log_q_rev - log_q_fwd)
    mh_err = float(np.max(np.abs(log_alpha - ref)))
    assert mh_err < 1e-9, f"MH ratio check failed: {mh_err}"
    print(f"selftest PASS grad_err={grad_err:.3e} mh_err={mh_err:.3e}")


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


def potential_and_grad(x_state, data_x, data_y, prior, chunk_rows):
    # U and gradU share the X @ theta matmul; one chunked pass computes both.
    flat = x_state.reshape((-1, x_state.shape[-1]))
    grad_out = cp.empty_like(flat)
    u_out = cp.empty(flat.shape[0], dtype=cp.float32)
    prior_mask = cp.ones(flat.shape[-1], dtype=cp.float32)
    prior_mask[0] = 0.0
    n = data_x.shape[0]
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        part = flat[start:stop]
        z = part @ data_x.T
        softplus = cp.maximum(z, 0.0) + cp.log1p(cp.exp(-cp.abs(z)))
        u = cp.mean(softplus - data_y[None, :] * z, axis=1)
        u = u + 0.5 * prior * cp.sum((part * prior_mask[None, :]) ** 2, axis=1)
        grad = (sigmoid(z) - data_y[None, :]) @ data_x / n
        grad = grad + prior * part * prior_mask[None, :]
        grad_out[start:stop] = grad
        u_out[start:stop] = u
    return u_out.reshape(x_state.shape[:-1]), grad_out.reshape(x_state.shape)


def predictive_logloss(estimates, data_x, data_y, chunk_rows=2048):
    losses = []
    eps = 1e-6
    for start in range(0, estimates.shape[0], chunk_rows):
        stop = min(start + chunk_rows, estimates.shape[0])
        probs = sigmoid(estimates[start:stop] @ data_x.T)
        loss = -(data_y[None, :] * cp.log(probs + eps) + (1 - data_y[None, :]) * cp.log(1 - probs + eps))
        losses.append(cp.mean(loss, axis=1))
    return cp.concatenate(losses)


def step_kernel(x, u_x, g_x, noise, design, data_x, data_y, chunk_rows, accept_acc, diag=None):
    # One transition; returns updated (x, U(x), gradU(x)). ULA never rejects.
    eta = design["eta"]
    sqrt_2eta = math.sqrt(2.0 * eta)
    y = x - eta * g_x + sqrt_2eta * noise
    if design["kernel"] == "ula":
        u_y, g_y = potential_and_grad(y, data_x, data_y, design["prior"], chunk_rows)
        accept_acc += 1.0
        return y, u_y, g_y
    u_y, g_y = potential_and_grad(y, data_x, data_y, design["prior"], chunk_rows)
    fwd = cp.sum((y - x + eta * g_x) ** 2, axis=-1)
    rev = cp.sum((x - y + eta * g_y) ** 2, axis=-1)
    log_alpha = u_x - u_y + (fwd - rev) / (4.0 * eta)
    # Accept indicators are per-chain independent uniforms; under coupled proposal
    # noise they become dependent across chains, which is exactly what we probe.
    accept = cp.log(cp.random.random(x.shape[:-1], dtype=cp.float32)) < log_alpha
    accept_acc += accept.mean()
    if diag is not None:
        # Rejection-leak diagnostic: count rejects and their per-coordinate
        # mean-square proposal noise (selection factor; 1 under no selection).
        reject = ~accept
        diag[0] += reject.sum()
        diag[1] += cp.where(reject, (noise * noise).mean(axis=-1), 0.0).sum()
        diag[2] += (noise * noise).mean()
    acc3 = accept[..., None]
    x_new = cp.where(acc3, y, x)
    u_new = cp.where(accept, u_y, u_x)
    g_new = cp.where(acc3, g_y, g_x)
    return x_new, u_new, g_new


def reference_mean(design, data_x, data_y, chunk_rows):
    cp.random.seed(design["seed"] + 77717)
    x = cp.zeros((design["ref_reps"], design["ref_chains"], design["dim"]), dtype=cp.float32)
    u, g = potential_and_grad(x, data_x, data_y, design["prior"], chunk_rows)
    acc = cp.zeros(design["dim"], dtype=cp.float32)
    accept_acc = cp.zeros((), dtype=cp.float32)
    windows = 0
    for step in range(design["ref_steps"]):
        noise = cp.random.standard_normal(x.shape, dtype=cp.float32)
        x, u, g = step_kernel(x, u, g, noise, design, data_x, data_y, chunk_rows, accept_acc)
        if step >= design["ref_burnin"]:
            acc += x.mean(axis=(0, 1))
            windows += 1
    return acc / windows


def run_method(design, data_x, data_y, test_x, test_y, ref, method, chunk_rows):
    kernel_offset = 7919 if design["kernel"] == "mala" else 0
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1) + kernel_offset)
    reps = design["reps"]
    chains = design["chains"]
    dim = design["dim"]
    x = cp.zeros((reps, chains, dim), dtype=cp.float32)
    ens_acc = cp.zeros((reps, dim), dtype=cp.float32)
    single_acc = cp.zeros((reps, dim), dtype=cp.float32)
    probe = normalized_probe(dim, design["design_seed"] + 1729)
    windows = 0
    warm = x[: min(reps, 2)].copy()
    warm_u, warm_g = potential_and_grad(warm, data_x, data_y, design["prior"], chunk_rows)
    warm_acc = cp.zeros((), dtype=cp.float32)
    warm, warm_u, warm_g = step_kernel(warm, warm_u, warm_g, make_noise(warm.shape, method), design, data_x, data_y, chunk_rows, warm_acc)
    _ = predictive_logloss(warm.mean(axis=1), test_x, test_y)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    u, g = potential_and_grad(x, data_x, data_y, design["prior"], chunk_rows)
    accept_acc = cp.zeros((), dtype=cp.float32)
    diag = cp.zeros(3, dtype=cp.float64) if design["kernel"] == "mala" else None
    for step in range(design["steps"]):
        x, u, g = step_kernel(x, u, g, make_noise(x.shape, method), design, data_x, data_y, chunk_rows, accept_acc, diag)
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
        "target": "real_logistic_mala_check",
        "dataset": design["dataset"],
        "method": method,
        "coupling": method,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "elapsed_sec_per_ensemble": elapsed / reps,
        "n_ensembles": reps,
        "windows": windows,
        "accept_rate": float((accept_acc / design["steps"]).get()),
        "reject_count": float(diag[0].get()) if diag is not None else 0.0,
        "noise_sq_given_reject": float((diag[1] / cp.maximum(diag[0], 1.0)).get()) if diag is not None else float("nan"),
        "noise_sq_overall": float((diag[2] / design["steps"]).get()) if diag is not None else float("nan"),
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


def calibrate_eta(dataset, dim, prior, data_x, data_y, chunk_rows, pilot_steps=100, pilot_reps=8, pilot_chains=8):
    # Deterministic per (dataset, prior): the chosen eta is shared by all methods
    # and both kernels in this cell, across separate invocations/replicates.
    base = default_eta(prior)
    seed = {"wdbc": 710000, "spambase": 720000, "adult": 730000}[dataset] + int(round(prior * 1000))
    chosen, chosen_rate, rates = None, None, {}
    for scale in ETA_SCAN:
        eta = scale * base
        design = {"kernel": "mala", "eta": eta, "prior": prior}
        cp.random.seed(seed + int(round(scale * 1000)))
        x = cp.zeros((pilot_reps, pilot_chains, dim), dtype=cp.float32)
        u, g = potential_and_grad(x, data_x, data_y, prior, chunk_rows)
        accept_acc = cp.zeros((), dtype=cp.float32)
        for _ in range(pilot_steps):
            noise = cp.random.standard_normal(x.shape, dtype=cp.float32)
            x, u, g = step_kernel(x, u, g, noise, design, data_x, data_y, chunk_rows, accept_acc)
        rate = float((accept_acc / pilot_steps).get())
        rates[scale] = rate
        if chosen is None and 0.4 <= rate <= 0.8:
            chosen, chosen_rate = eta, rate
    if chosen is None:
        # No scanned eta landed in [0.4, 0.8]: take the one closest to 0.6.
        scale = min(rates, key=lambda s: abs(rates[s] - 0.6))
        chosen, chosen_rate = scale * base, rates[scale]
    print(json.dumps({"calibrate": dataset, "prior": prior, "rates": {str(k): v for k, v in sorted(rates.items())}, "eta": chosen}, sort_keys=True), flush=True)
    return chosen, chosen_rate


def make_designs(dataset, dim, args, eta_by_prior, pilot_by_prior):
    defaults = DATASET_DEFAULTS[dataset]
    chains = args.chains or [8]
    priors = args.priors or [0.1, 0.3]
    seed_base = {"wdbc": 710000, "spambase": 720000, "adult": 730000}[dataset]
    idx = 0
    for prior in priors:
        eta = eta_by_prior[prior]
        for k in chains:
            reps = args.reps or max(24, defaults["reps"] // max(1, k // 8))
            yield {
                "dataset": dataset,
                "dim": dim,
                "chains": k,
                "prior": prior,
                "eta": eta,
                "eta_calibrated": bool(args.calibrate),
                "pilot_accept": pilot_by_prior[prior],
                "kernel": args.kernel,
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
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--kernel", choices=["ula", "mala"], default="mala")
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
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.out is None:
        parser.error("--out is required unless --selftest")
    args.out.parent.mkdir(parents=True, exist_ok=True)
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
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    with args.out.open("a", encoding="utf-8") as f:
        for dataset in datasets:
            data_x, data_y, test_x, test_y = load_dataset(args.data_dir, dataset)
            dim = int(data_x.shape[1])
            priors = args.priors or [0.1, 0.3]
            eta_by_prior, pilot_by_prior = {}, {}
            for prior in priors:
                if args.eta is not None:
                    eta_by_prior[prior], pilot_by_prior[prior] = args.eta, None
                elif args.calibrate:
                    eta_by_prior[prior], pilot_by_prior[prior] = calibrate_eta(dataset, dim, prior, data_x, data_y, args.chunk_rows)
                else:
                    eta_by_prior[prior], pilot_by_prior[prior] = default_eta(prior), None
            designs = list(make_designs(dataset, dim, args, eta_by_prior, pilot_by_prior))
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
