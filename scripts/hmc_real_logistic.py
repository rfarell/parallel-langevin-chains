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
# 300 outer steps at L=10 leapfrog gradients each ~ 3000 gradient evaluations
# per chain, comparable to the ULA/MALA campaigns.
DATASET_DEFAULTS = {
    "wdbc": {"steps": 300, "burnin": 150, "ref_steps": 600, "ref_burnin": 300, "reps": 128},
    "spambase": {"steps": 300, "burnin": 150, "ref_steps": 600, "ref_burnin": 300, "reps": 96},
    "adult": {"steps": 300, "burnin": 150, "ref_steps": 600, "ref_burnin": 300, "reps": 48},
}
LEAPFROG_STEPS = 10
# Geometric grid (ratio 1/sqrt(2)), scanned largest-first: the pilot picks the
# largest eps with realized acceptance in [0.6, 0.9].  The top (6.4) rejects
# essentially everything on all cells, so the band is bracketed within the grid.
EPS_SCAN = [6.4 * 2.0 ** (-k / 2.0) for k in range(18)]
DATASET_LABELS = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def np_potential_grad(theta, data_x, data_y, prior):
    # CPU mirror of the GPU potential for the selftest.
    z = theta @ data_x.T
    u_data = np.mean(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - data_y[None, :] * z, axis=-1)
    mask = np.ones(theta.shape[-1])
    mask[0] = 0.0
    u = u_data + 0.5 * prior * np.sum((theta * mask[None, :]) ** 2, axis=-1)
    probs = 1.0 / (1.0 + np.exp(-z))
    grad = (probs - data_y[None, :]) @ data_x / data_x.shape[0] + prior * theta * mask[None, :]
    return u, grad


def np_leapfrog(x, p, eps, n_leapfrog, data_x, data_y, prior):
    # CPU mirror of the GPU trajectory: half momentum steps at the ends.
    _, g = np_potential_grad(x, data_x, data_y, prior)
    p = p - 0.5 * eps * g
    for l in range(n_leapfrog):
        x = x + eps * p
        _, g = np_potential_grad(x, data_x, data_y, prior)
        if l + 1 < n_leapfrog:
            p = p - eps * g
    p = p - 0.5 * eps * g
    return x, p


def selftest():
    rng = np.random.default_rng(0)
    n, d, prior = 32, 5, 0.3
    data_x = rng.standard_normal((n, d))
    data_x[:, 0] = 1.0
    data_y = (rng.random(n) < 0.5).astype(np.float64)
    x = rng.standard_normal((3, d)) * 0.5
    p = rng.standard_normal((3, d))
    u0, _ = np_potential_grad(x, data_x, data_y, prior)
    h0 = u0 + 0.5 * np.sum(p * p, axis=-1)
    # Leapfrog energy error must vanish as eps -> 0 (at least second order).
    eps_grid = [0.4, 0.2, 0.1, 0.05, 0.025]
    errs = []
    for eps in eps_grid:
        x1, p1 = np_leapfrog(x, p, eps, LEAPFROG_STEPS, data_x, data_y, prior)
        u1, _ = np_potential_grad(x1, data_x, data_y, prior)
        h1 = u1 + 0.5 * np.sum(p1 * p1, axis=-1)
        errs.append(float(np.max(np.abs(h1 - h0))))
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), f"energy error not decreasing: {errs}"
    slope = float(np.polyfit(np.log(eps_grid), np.log(errs), 1)[0])
    assert slope > 1.8, f"energy-error order too low: slope={slope:.3f} errs={errs}"
    assert errs[-1] < 1e-4, f"energy error too large at eps={eps_grid[-1]}: {errs[-1]:.3e}"
    # MH log-ratio from total energy must match the explicit joint density
    # ratio pi(x')N(p'|0,I) / pi(x)N(p|0,I); leapfrog is volume-preserving and
    # reversible so no Jacobian or proposal-density terms appear.
    eps = 0.2
    x1, p1 = np_leapfrog(x, p, eps, LEAPFROG_STEPS, data_x, data_y, prior)
    u1, _ = np_potential_grad(x1, data_x, data_y, prior)
    h1 = u1 + 0.5 * np.sum(p1 * p1, axis=-1)
    log_alpha = h0 - h1

    def joint_log_density(u, mom):
        return -u - 0.5 * np.sum(mom * mom, axis=-1) - 0.5 * d * math.log(2.0 * math.pi)

    ref = joint_log_density(u1, p1) - joint_log_density(u0, p)
    mh_err = float(np.max(np.abs(log_alpha - ref)))
    assert mh_err < 1e-9, f"MH ratio check failed: {mh_err}"
    # Reversibility: integrating back with negated momentum recovers the start.
    x2, p2 = np_leapfrog(x1, -p1, eps, LEAPFROG_STEPS, data_x, data_y, prior)
    rev_err = float(max(np.max(np.abs(x2 - x)), np.max(np.abs(-p2 - p))))
    assert rev_err < 1e-9, f"reversibility check failed: {rev_err}"
    print(
        f"selftest PASS energy_errs={['%.3e' % e for e in errs]} slope={slope:.2f} "
        f"mh_err={mh_err:.3e} rev_err={rev_err:.3e}"
    )


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


def hmc_step(x, u_x, g_x, p0, design, data_x, data_y, chunk_rows, accept_acc, diag=None):
    # One outer HMC step: momentum refresh p0 (the only injected randomness),
    # L leapfrog steps, then per-chain MH on the total energy H = U + |p|^2/2.
    eps = design["eps"]
    n_leapfrog = design["leapfrog"]
    h_old = u_x + 0.5 * cp.sum(p0 * p0, axis=-1)
    x_new = x
    p = p0 - 0.5 * eps * g_x
    for l in range(n_leapfrog):
        x_new = x_new + eps * p
        u_new, g_new = potential_and_grad(x_new, data_x, data_y, design["prior"], chunk_rows)
        if l + 1 < n_leapfrog:
            p = p - eps * g_new
    p = p - 0.5 * eps * g_new
    h_new = u_new + 0.5 * cp.sum(p * p, axis=-1)
    # NaN/inf energies (divergent trajectories) fail the comparison and reject.
    log_alpha = h_old - h_new
    # Accept indicators are per-chain independent uniforms; under coupled
    # momentum refreshes they become dependent across chains (the leak we probe).
    accept = cp.log(cp.random.random(x.shape[:-1], dtype=cp.float32)) < log_alpha
    accept_acc += accept.mean()
    if diag is not None:
        # Rejection-leak diagnostic: count rejects and their per-coordinate
        # mean-square refreshed momentum (selection factor; 1 under no selection).
        reject = ~accept
        diag[0] += reject.sum()
        diag[1] += cp.where(reject, (p0 * p0).mean(axis=-1), 0.0).sum()
        diag[2] += (p0 * p0).mean()
    acc3 = accept[..., None]
    x_out = cp.where(acc3, x_new, x)
    u_out = cp.where(accept, u_new, u_x)
    g_out = cp.where(acc3, g_new, g_x)
    return x_out, u_out, g_out


def reference_mean(design, data_x, data_y, chunk_rows):
    cp.random.seed(design["seed"] + 77717)
    x = cp.zeros((design["ref_reps"], design["ref_chains"], design["dim"]), dtype=cp.float32)
    u, g = potential_and_grad(x, data_x, data_y, design["prior"], chunk_rows)
    acc = cp.zeros(design["dim"], dtype=cp.float32)
    accept_acc = cp.zeros((), dtype=cp.float32)
    windows = 0
    for step in range(design["ref_steps"]):
        p0 = cp.random.standard_normal(x.shape, dtype=cp.float32)
        x, u, g = hmc_step(x, u, g, p0, design, data_x, data_y, chunk_rows, accept_acc)
        if step >= design["ref_burnin"]:
            acc += x.mean(axis=(0, 1))
            windows += 1
    return acc / windows


def run_method(design, data_x, data_y, test_x, test_y, ref, method, chunk_rows):
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1) + 40009)
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
    warm, warm_u, warm_g = hmc_step(warm, warm_u, warm_g, make_noise(warm.shape, method), design, data_x, data_y, chunk_rows, warm_acc)
    _ = predictive_logloss(warm.mean(axis=1), test_x, test_y)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    u, g = potential_and_grad(x, data_x, data_y, design["prior"], chunk_rows)
    accept_acc = cp.zeros((), dtype=cp.float32)
    diag = cp.zeros(3, dtype=cp.float64)
    for step in range(design["steps"]):
        x, u, g = hmc_step(x, u, g, make_noise(x.shape, method), design, data_x, data_y, chunk_rows, accept_acc, diag)
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
        "target": "real_logistic_hmc",
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
        "reject_count": float(diag[0].get()),
        "mom_sq_given_reject": float((diag[1] / cp.maximum(diag[0], 1.0)).get()),
        "mom_sq_overall": float((diag[2] / design["steps"]).get()),
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


def calibrate_eps(dataset, dim, prior, leapfrog, data_x, data_y, chunk_rows, pilot_steps=60, pilot_reps=8, pilot_chains=8):
    # Deterministic per (dataset, prior): the chosen eps is shared by all
    # couplings in this cell, across separate invocations/replicates.
    seed = {"wdbc": 710000, "spambase": 720000, "adult": 730000}[dataset] + int(round(prior * 1000)) + 424243
    chosen, chosen_rate, rates = None, None, {}
    for gi, eps in enumerate(EPS_SCAN):
        design = {"eps": eps, "leapfrog": leapfrog, "prior": prior}
        cp.random.seed(seed + 1000 * gi)
        x = cp.zeros((pilot_reps, pilot_chains, dim), dtype=cp.float32)
        u, g = potential_and_grad(x, data_x, data_y, prior, chunk_rows)
        accept_acc = cp.zeros((), dtype=cp.float32)
        for _ in range(pilot_steps):
            p0 = cp.random.standard_normal(x.shape, dtype=cp.float32)
            x, u, g = hmc_step(x, u, g, p0, design, data_x, data_y, chunk_rows, accept_acc)
        rate = float((accept_acc / pilot_steps).get())
        rates[eps] = rate
        if chosen is None and 0.6 <= rate <= 0.9:
            # EPS_SCAN is descending, so the first hit is the largest eps in band.
            chosen, chosen_rate = eps, rate
    if chosen is None:
        eps = min(rates, key=lambda e: abs(rates[e] - 0.75))
        chosen, chosen_rate = eps, rates[eps]
    print(json.dumps({"calibrate": dataset, "prior": prior, "leapfrog": leapfrog, "rates": {f"{k:.6g}": v for k, v in sorted(rates.items())}, "eps": chosen, "pilot_accept": chosen_rate}, sort_keys=True), flush=True)
    return chosen, chosen_rate


def make_designs(dataset, dim, args, eps_by_prior, pilot_by_prior):
    defaults = DATASET_DEFAULTS[dataset]
    chains = args.chains or [8]
    priors = args.priors or [0.1, 0.3]
    seed_base = {"wdbc": 710000, "spambase": 720000, "adult": 730000}[dataset]
    idx = 0
    for prior in priors:
        eps = eps_by_prior[prior]
        for k in chains:
            reps = args.reps or max(24, defaults["reps"] // max(1, k // 8))
            yield {
                "dataset": dataset,
                "dim": dim,
                "chains": k,
                "prior": prior,
                "eps": eps,
                "leapfrog": args.leapfrog,
                "eps_calibrated": args.eps is None,
                "pilot_accept": pilot_by_prior[prior],
                "kernel": "hmc",
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


def latex_sci(x, sig=3):
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def fmt(x, sig=3):
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return latex_sci(x, sig)
    decimals = max(sig - 1 - int(np.floor(np.log10(abs(x)))), 0)
    return f"{x:.{decimals}f}"


def cell(vals, sig=3):
    med = float(np.median(vals))
    return f"{fmt(med, sig)} [{fmt(min(vals), sig)}, {fmt(max(vals), sig)}]"


def ratios(rows, key, num_method, field="trace_var", den_method="iid"):
    sub = [r for r in rows if all(r.get(k) == v for k, v in key.items())]
    out = []
    for rep in sorted({r["replicate"] for r in sub}):
        num = [r[field] for r in sub if r["replicate"] == rep and r["method"] == num_method]
        den = [r[field] for r in sub if r["replicate"] == rep and r["method"] == den_method]
        assert len(num) == 1 and len(den) == 1, f"expected 1 row/method, got {key} rep{rep}: {len(num)}/{len(den)}"
        out.append(num[0] / den[0])
    assert len(out) == 3, f"expected 3 replicates for {key}, got {len(out)}"
    return out


def absolutes(rows, key, method, field):
    sub = [r[field] for r in rows if all(r.get(k) == v for k, v in key.items()) and r["method"] == method]
    assert len(sub) == 3, f"expected 3 replicates for {key} {method} {field}, got {len(sub)}"
    return sub


def make_table(table_inputs, table_out):
    rows = []
    for path in table_inputs:
        with Path(path).open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & prior & HMC accept & $C_\\star$/iid & pairwise/iid & pred.\\ $p\\,s$ \\\\",
        "\\midrule",
    ]
    summary = ["== hmc_real.tex (trace-var ratios, median across 3 reps) =="]
    for ds in DATASETS:
        for prior in [0.1, 0.3]:
            key = {"dataset": ds, "prior": prior}
            zs = ratios(rows, key, "zero_sum")
            pa = ratios(rows, key, "antithetic")
            accept = absolutes(rows, key, "zero_sum", "accept_rate")
            p_rej = [1.0 - a for a in accept]
            s_sel = [r["mom_sq_given_reject"] / r["mom_sq_overall"]
                     for r in rows if r["dataset"] == ds and r["prior"] == prior and r["method"] == "zero_sum"]
            assert len(s_sel) == 3
            ps = [p * s for p, s in zip(p_rej, s_sel)]
            acc = f"{float(np.median(accept)):.2f}"
            lines.append(f"{DATASET_LABELS[ds]} & {prior} & {acc} & {cell(zs)} & {cell(pa)} & {float(np.median(ps)):.3f} \\\\")
            summary.append(
                f"  {ds} prior={prior}: accept={acc} C*/iid={np.median(zs):.3g} pair/iid={np.median(pa):.3g} "
                f"p={np.median(p_rej):.3f} s={np.median(s_sel):.3f} pred ps={np.median(ps):.3g}"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    Path(table_out).write_text("\n".join(lines), encoding="utf-8")
    summary.append(f"  -> wrote {table_out}")
    print("\n".join(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
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
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--leapfrog", type=int, default=LEAPFROG_STEPS)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--make-table", action="store_true")
    parser.add_argument("--table-inputs", nargs="+", type=Path,
                        default=[Path(f"outputs/sweeps/hmc_real_rep{r}.jsonl") for r in range(3)])
    parser.add_argument("--table-out", type=Path, default=Path("paper/tables/hmc_real.tex"))
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.make_table:
        make_table(args.table_inputs, args.table_out)
        return
    if args.out is None:
        parser.error("--out is required unless --selftest/--make-table")
    if args.out.exists():
        parser.error(f"refusing to overwrite existing {args.out}")
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
    with args.out.open("x", encoding="utf-8") as f:
        for dataset in datasets:
            data_x, data_y, test_x, test_y = load_dataset(args.data_dir, dataset)
            dim = int(data_x.shape[1])
            priors = args.priors or [0.1, 0.3]
            eps_by_prior, pilot_by_prior = {}, {}
            for prior in priors:
                if args.eps is not None:
                    eps_by_prior[prior], pilot_by_prior[prior] = args.eps, None
                else:
                    eps_by_prior[prior], pilot_by_prior[prior] = calibrate_eps(
                        dataset, dim, prior, args.leapfrog, data_x, data_y, args.chunk_rows)
            designs = list(make_designs(dataset, dim, args, eps_by_prior, pilot_by_prior))
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
