"""Diagonal-preconditioned ULA on real logistic-regression posteriors.

Verifies that the zero-sum coupling's variance reduction carries over to a
FIXED (non-adaptive) diagonally preconditioned kernel

    x <- x - eta * P grad U(x) + sqrt(2*eta) * P^{1/2} xi,

which is plain ULA after the coordinate change y = P^{-1/2} x, so the
paper's zero-sum identity should transfer exactly; this script checks that
empirically on wdbc/spambase/adult.

Preconditioner (fixed per dataset x prior cell, computed from train data):

    P = diag(1 / (0.25 * colmean(X^2) + prior))

i.e. the inverse Jacobi/Gauss-Newton diagonal curvature bound at theta = 0:
the logistic Hessian is X^T S X / n with S = diag(p(1-p)) <= I/4, so its
diagonal is bounded by 0.25 * colmean(X^2); `prior` adds the ridge
curvature. The intercept column (all ones) is included in colmean(X^2) and
also receives the +prior damping even though the prior in U excludes the
intercept -- this keeps P strictly positive with a uniform construction.
The campaign datasets are standardized (colmean(X^2) ~= 1 per column), so P
is close to (0.25 + prior)^{-1} I here; the run is still a genuine
coordinate change of the parent kernel and is documented as such.

Step size: per cell, a short iid pilot (100 steps, 8 ensembles x K chains)
picks the largest eta from the geometric grid 0.5 * 2^{-3..3} with no
divergence (running max |x| < 1e3). The chosen eta is deterministic per
cell, shared by all methods, replicates, and the reference run, and logged.

Noise coupling: xi is drawn by make_noise (copied from the parent), i.e.
centered across chains BEFORE the P^{1/2} scaling, so per-chain marginals
stay N(0, P). Everything else (loading, metrics, seeds, JSONL, --quick,
warmup timing) mirrors scripts/real_logistic_gpu.py.
"""

import argparse
import json
import math
import os
import subprocess
import sys
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
SEED_BASE = {"wdbc": 910000, "spambase": 920000, "adult": 930000}
# Geometric grid around 0.5 (factor 2, three octaves each way), scanned
# descending; the pilot takes the largest entry with no divergence.
ETA_GRID = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
DATASET_LABELS = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}


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


def normalized_probe(dim, seed):
    rng = cp.random.RandomState(seed)
    probe = rng.standard_normal(dim, dtype=cp.float32)
    probe[0] = 0.0
    return probe / cp.linalg.norm(probe)


def precond_diag(data_x, prior):
    """P = diag(1/(0.25*colmean(X^2) + prior)); see module docstring."""
    return 1.0 / (0.25 * cp.mean(data_x * data_x, axis=0) + prior)


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


def pilot_eta(dataset, dim, prior, p_diag, data_x, data_y, chunk_rows, steps=100, reps=8, chains=8):
    """Largest ETA_GRID entry with running max |x| < 1e3 over an iid pilot.

    Deterministic per (dataset, prior): the chosen eta is shared by all
    methods and replicates in this cell, across separate invocations.
    """
    seed = SEED_BASE[dataset] + int(round(prior * 1000))
    p = p_diag[None, None, :]
    sqrt_p = cp.sqrt(p_diag)[None, None, :]
    maxabs = {}
    chosen = None
    for eta in ETA_GRID:
        cp.random.seed(seed + int(round(eta * 10000)))
        x = cp.zeros((reps, chains, dim), dtype=cp.float32)
        run_max = cp.zeros((), dtype=cp.float32)
        sqrt_2eta = math.sqrt(2.0 * eta)
        for _ in range(steps):
            grad = logistic_grad(x, data_x, data_y, prior, chunk_rows)
            x = x - eta * p * grad + sqrt_2eta * sqrt_p * cp.random.standard_normal(x.shape, dtype=cp.float32)
            run_max = cp.maximum(run_max, cp.max(cp.abs(x)))
        m = float(run_max.get())
        maxabs[eta] = m
        if chosen is None and math.isfinite(m) and m < 1e3:
            chosen = eta
    if chosen is None:
        raise RuntimeError(f"pilot found no stable eta for {dataset} prior={prior}: {maxabs}")
    print(json.dumps({"pilot": dataset, "prior": prior, "maxabs": {str(k): v for k, v in maxabs.items()}, "eta": chosen}, sort_keys=True), flush=True)
    return chosen, maxabs[chosen]


def reference_mean(design, data_x, data_y, p_diag, chunk_rows):
    cp.random.seed(design["seed"] + 77717)
    x = cp.zeros((design["ref_reps"], design["ref_chains"], design["dim"]), dtype=cp.float32)
    p = p_diag[None, None, :]
    sqrt_p = cp.sqrt(p_diag)[None, None, :]
    acc = cp.zeros(design["dim"], dtype=cp.float32)
    sqrt_2eta = math.sqrt(2.0 * design["eta"])
    windows = 0
    for step in range(design["ref_steps"]):
        grad = logistic_grad(x, data_x, data_y, design["prior"], chunk_rows)
        x = x - design["eta"] * p * grad + sqrt_2eta * sqrt_p * cp.random.standard_normal(x.shape, dtype=cp.float32)
        if step >= design["ref_burnin"]:
            acc += x.mean(axis=(0, 1))
            windows += 1
    return acc / windows


def run_method(design, data_x, data_y, test_x, test_y, ref, p_diag, method, chunk_rows):
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1))
    reps = design["reps"]
    chains = design["chains"]
    dim = design["dim"]
    x = cp.zeros((reps, chains, dim), dtype=cp.float32)
    p = p_diag[None, None, :]
    sqrt_p = cp.sqrt(p_diag)[None, None, :]
    ens_acc = cp.zeros((reps, dim), dtype=cp.float32)
    single_acc = cp.zeros((reps, dim), dtype=cp.float32)
    probe = normalized_probe(dim, design["design_seed"] + 1729)
    sqrt_2eta = math.sqrt(2.0 * design["eta"])
    windows = 0
    warm = x[: min(reps, 2)].copy()
    warm_grad = logistic_grad(warm, data_x, data_y, design["prior"], chunk_rows)
    warm = warm - design["eta"] * p * warm_grad + sqrt_2eta * sqrt_p * make_noise(warm.shape, method)
    _ = predictive_logloss(warm.mean(axis=1), test_x, test_y)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for step in range(design["steps"]):
        grad = logistic_grad(x, data_x, data_y, design["prior"], chunk_rows)
        # xi from make_noise is centered across chains BEFORE the P^{1/2}
        # scaling, so per-chain marginals stay N(0, P).
        x = x - design["eta"] * p * grad + sqrt_2eta * sqrt_p * make_noise(x.shape, method)
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
        "target": "real_logistic_preconditioned",
        "dataset": design["dataset"],
        "method": method,
        "coupling": method,
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


def make_designs(dataset, dim, args, cell_info):
    defaults = DATASET_DEFAULTS[dataset]
    chains = args.chains or [8]
    priors = args.priors or [0.1, 0.3]
    seed_base = SEED_BASE[dataset]
    idx = 0
    for prior in priors:
        info = cell_info[prior]
        for k in chains:
            reps = args.reps or max(24, defaults["reps"] // max(1, k // 8))
            yield {
                "dataset": dataset,
                "dim": dim,
                "chains": k,
                "prior": prior,
                "eta": info["eta"],
                "eta_source": info["eta_source"],
                "pilot_maxabs": info["pilot_maxabs"],
                "precond": "jacobi_gn_theta0",
                "precond_min": info["precond_min"],
                "precond_max": info["precond_max"],
                "precond_intercept": info["precond_intercept"],
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


# ---------------------------------------------------------------------------
# Table generation (house style; helpers cloned from summarize_new_experiments)


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


def make_table(sweeps_dir, tables_dir):
    rows = []
    for rep in (0, 1, 2):
        path = sweeps_dir / f"preconditioned_real_rep{rep}.jsonl"
        with path.open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    lines = [
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Dataset & prior & $C_\\star$/iid & pairwise/iid \\\\",
        "\\midrule",
    ]
    print("== preconditioned_real.tex (trace-var ratios, median across 3 reps) ==")
    for ds in DATASETS:
        for prior in [0.1, 0.3]:
            key = {"dataset": ds, "prior": prior, "chains": 8}
            zs = ratios(rows, key, "zero_sum")
            pa = ratios(rows, key, "antithetic")
            eta = {r["eta"] for r in rows if r["dataset"] == ds and r["prior"] == prior}
            lines.append(f"{DATASET_LABELS[ds]} & {prior} & {cell(zs)} & {cell(pa)} \\\\")
            print(f"  {ds} prior={prior}: eta={sorted(eta)} C*/iid={np.median(zs):.3g} pair/iid={np.median(pa):.3g}")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path = tables_dir / "preconditioned_real.tex"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> wrote {out_path}")


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
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--make-table", action="store_true")
    parser.add_argument("--sweeps-dir", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--tables-dir", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()
    if args.make_table:
        args.tables_dir.mkdir(parents=True, exist_ok=True)
        make_table(args.sweeps_dir, args.tables_dir)
        return
    if args.out is None:
        parser.error("--out is required unless --make-table")
    if args.out.exists():
        sys.exit(f"refusing to append: {args.out} already exists")
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
            cell_info, p_by_prior = {}, {}
            for prior in priors:
                p_diag = precond_diag(data_x, prior)
                p_by_prior[prior] = p_diag
                if args.eta is not None:
                    eta, eta_source, pilot_max = args.eta, "cli", None
                else:
                    eta, pilot_max = pilot_eta(dataset, dim, prior, p_diag, data_x, data_y, args.chunk_rows)
                    eta_source = "pilot"
                cell_info[prior] = {
                    "eta": eta,
                    "eta_source": eta_source,
                    "pilot_maxabs": pilot_max,
                    "precond_min": float(p_diag.min().get()),
                    "precond_max": float(p_diag.max().get()),
                    "precond_intercept": float(p_diag[0].get()),
                }
            designs = list(make_designs(dataset, dim, args, cell_info))
            if args.limit is not None:
                designs = designs[: args.limit]
            for design in designs:
                p_diag = p_by_prior[design["prior"]]
                ref = reference_mean(design, data_x, data_y, p_diag, args.chunk_rows)
                for method in args.methods:
                    result = run_method(design, data_x, data_y, test_x, test_y, ref, p_diag, method, args.chunk_rows)
                    f.write(json.dumps(result, sort_keys=True) + "\n")
                    f.flush()
                    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
