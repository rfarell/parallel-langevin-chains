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
DATASET_DEFAULTS = {
    "wdbc": {"steps": 900, "burnin": 450, "reps": 256},
    "spambase": {"steps": 900, "burnin": 450, "reps": 192},
    "adult": {"steps": 700, "burnin": 350, "reps": 64},
}
TPRIOR_SEED_BASE = {"wdbc": 810000, "spambase": 820000, "adult": 830000}
HIER_SEED_BASE = 840000
ADULT_COLUMNS = 15  # columns per row in data/raw/adult.data
ADULT_EDU_COL = 3


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
    y_train = cp.asarray(data["y_train"], dtype=cp.float32)
    return x_train, y_train


def tprior_grad(x_state, data_x, data_y, df, s2, chunk_rows):
    # -grad log t-prior per coordinate: (df+1)*theta / (df*s2 + theta^2); intercept unpenalized
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
        grad = grad + (df + 1.0) * part / (df * s2 + part * part) * prior_mask[None, :]
        out[start:stop] = grad
    return out.reshape(x_state.shape)


def parse_adult_groups(raw_path):
    # replicate download_real_data.py's filter exactly: skip blank/'|' lines,
    # skip rows with wrong field count or any '?' value; train rows align with adult.npz X_train
    groups = []
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|"):
                continue
            values = [v.strip() for v in line.split(",")]
            if len(values) != ADULT_COLUMNS or "?" in values:
                continue
            groups.append(values[ADULT_EDU_COL])
    levels = sorted(set(groups))
    idx = np.asarray([levels.index(g) for g in groups], dtype=np.int64)
    return idx, levels


def load_hierarchical(data_dir, raw_dir):
    data = np.load(data_dir / "adult.npz", allow_pickle=True)
    names = [str(v) for v in data["feature_names"]]
    # education one-hot columns are replaced by the random intercepts u_g
    keep = np.asarray([not v.startswith("education=") for v in names], dtype=bool)
    x_np = add_intercept(data["X_train"][:, keep])
    group_idx, levels = parse_adult_groups(raw_dir / "adult.data")
    if len(group_idx) != x_np.shape[0]:
        raise RuntimeError(f"raw/npz row mismatch: {len(group_idx)} vs {x_np.shape[0]}")
    n_groups = len(levels)
    gmat_np = np.zeros((x_np.shape[0], n_groups), dtype=np.float32)
    gmat_np[np.arange(x_np.shape[0]), group_idx] = 1.0
    data_x = cp.asarray(x_np, dtype=cp.float32)
    data_y = cp.asarray(data["y_train"], dtype=cp.float32)
    gmat = cp.asarray(gmat_np)
    return data_x, data_y, gmat, n_groups


def hier_grad(x_state, data_x, data_y, gmat, lam_beta, chunk_rows):
    # theta = [beta (d_beta, intercept first), u (G), rho (1)], tau = exp(rho)
    # U = (1/n) sum_i [log(1+exp(z_i)) - y_i z_i] + (lam_beta/2)||beta_masked||^2
    #     + sum_g u_g^2/(2 tau^2) + G*rho + rho^2/2,  z_i = x_i.beta + u_{g(i)}
    flat = x_state.reshape((-1, x_state.shape[-1]))
    out = cp.empty_like(flat)
    n, db = data_x.shape
    n_groups = gmat.shape[1]
    prior_mask = cp.ones(db, dtype=cp.float32)
    prior_mask[0] = 0.0
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        beta = flat[start:stop, :db]
        u = flat[start:stop, db:db + n_groups]
        rho = flat[start:stop, db + n_groups]
        z = beta @ data_x.T + u @ gmat.T
        resid = sigmoid(z) - data_y[None, :]
        g_beta = resid @ data_x / n + lam_beta * beta * prior_mask[None, :]
        inv_tau2 = cp.exp(-2.0 * cp.clip(rho, -45.0, 45.0))  # clip: float32 overflow guard only
        g_u = resid @ gmat / n + u * inv_tau2[:, None]
        g_rho = -cp.sum(u * u, axis=1) * inv_tau2 + n_groups + rho
        out[start:stop, :db] = g_beta
        out[start:stop, db:db + n_groups] = g_u
        out[start:stop, db + n_groups] = g_rho
    return out.reshape(x_state.shape)


def hier_energy_np(theta, x_np, y_np, group_idx, n_groups, lam_beta):
    db = x_np.shape[1]
    beta = theta[:db]
    u = theta[db:db + n_groups]
    rho = theta[db + n_groups]
    z = x_np @ beta + u[group_idx]
    nll = np.mean(np.logaddexp(0.0, z) - y_np * z)
    prior_beta = 0.5 * lam_beta * np.sum(beta[1:] ** 2)
    prior_u = 0.5 * np.sum(u * u) * np.exp(-2.0 * rho)
    return nll + prior_beta + prior_u + n_groups * rho + 0.5 * rho * rho


def selftest_hier(data_x, data_y, gmat, lam_beta, seed=20260712):
    rng = np.random.default_rng(seed)
    x_np = cp.asnumpy(data_x).astype(np.float64)
    y_np = cp.asnumpy(data_y).astype(np.float64)
    group_idx = np.argmax(cp.asnumpy(gmat), axis=1)
    sub = np.sort(rng.choice(x_np.shape[0], size=256, replace=False))
    x_np, y_np, group_idx = x_np[sub], y_np[sub], group_idx[sub]
    n_groups = gmat.shape[1]
    dim = x_np.shape[1] + n_groups + 1
    theta = 0.3 * rng.standard_normal(dim)
    x_sub = cp.asarray(x_np, dtype=cp.float32)
    y_sub = cp.asarray(y_np, dtype=cp.float32)
    gmat_sub = cp.zeros((len(sub), n_groups), dtype=cp.float32)
    gmat_sub[cp.arange(len(sub)), cp.asarray(group_idx)] = 1.0
    analytic = cp.asnumpy(
        hier_grad(cp.asarray(theta[None, None, :], dtype=cp.float32), x_sub, y_sub, gmat_sub, lam_beta, 64)
    ).ravel().astype(np.float64)
    h = 1e-6
    fd = np.empty(dim)
    for j in range(dim):
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        fd[j] = (hier_energy_np(tp, x_np, y_np, group_idx, n_groups, lam_beta)
                 - hier_energy_np(tm, x_np, y_np, group_idx, n_groups, lam_beta)) / (2.0 * h)
    err = np.max(np.abs(analytic - fd) / np.maximum(1e-3, np.abs(fd)))
    ok = err < 1e-3  # analytic path is float32; FD is float64
    print(json.dumps({"selftest": "hier_grad", "max_rel_err": float(err), "pass": bool(ok),
                      "dim": int(dim)}, sort_keys=True))
    return ok


def calibrate_eta(design, grad_fn, pilot_reps=8, check_every=25, max_halvings=3):
    eta = design["eta"]
    halvings = 0
    for attempt in range(max_halvings + 1):
        cp.random.seed(design["seed"] + 55501)
        x = cp.zeros((pilot_reps, design["chains"], design["dim"]), dtype=cp.float32)
        sqrt_2eta = math.sqrt(2.0 * eta)
        ok = True
        for step in range(design["steps"]):
            x = x - eta * grad_fn(x) + sqrt_2eta * make_noise(x.shape, "iid")
            if (step + 1) % check_every == 0 or step == design["steps"] - 1:
                norms = cp.sqrt(cp.sum(x * x, axis=2))
                worst = float(cp.max(norms).get())
                if not math.isfinite(worst) or worst > 1e3:
                    ok = False
                    break
        if ok:
            return eta, halvings, True
        if attempt < max_halvings:
            eta *= 0.5
            halvings += 1
    return eta, halvings, False


def run_method(design, grad_fn, method, beta_dim):
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1))
    reps = design["reps"]
    chains = design["chains"]
    dim = design["dim"]
    x = cp.zeros((reps, chains, dim), dtype=cp.float32)
    ens_acc = cp.zeros((reps, dim), dtype=cp.float32)
    sqrt_2eta = math.sqrt(2.0 * design["eta"])
    windows = 0
    warm = x[: min(reps, 2)].copy()
    warm = warm - design["eta"] * grad_fn(warm) + sqrt_2eta * make_noise(warm.shape, method)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for step in range(design["steps"]):
        grad = grad_fn(x)
        x = x - design["eta"] * grad + sqrt_2eta * make_noise(x.shape, method)
        if step >= design["burnin"]:
            ens_acc += x.mean(axis=1)
            windows += 1
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    ens = ens_acc / windows
    finite = cp.all(cp.isfinite(ens), axis=1) & cp.all(cp.isfinite(x.reshape((reps, -1))), axis=1)
    n_finite = int(cp.sum(finite).get())
    ens_f = ens[finite]
    x_f = x[finite]
    ens_centered = ens_f - ens_f.mean(axis=0)
    beta_centered = ens_centered[:, :beta_dim]
    chain0 = x_f[:, 0, :]
    ens_terminal = x_f.mean(axis=1)
    return {
        **design,
        "target": "nonlogconcave_real",
        "method": method,
        "coupling": method,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "elapsed_sec_per_ensemble": elapsed / reps,
        "n_ensembles": reps,
        "windows": windows,
        "beta_dim": beta_dim,
        "finite_frac": n_finite / reps,
        "trace_var": float(cp.mean(cp.sum(ens_centered * ens_centered, axis=1)).get()),
        "trace_var_beta": float(cp.mean(cp.sum(beta_centered * beta_centered, axis=1)).get()),
        "terminal_spread": float(cp.mean(cp.sum((x_f - ens_terminal[:, None, :]) ** 2, axis=2) / dim).get()),
        "marginal_second": float(cp.mean(cp.sum(chain0 * chain0, axis=1) / dim).get()),
    }


def run_tprior(args, out_file):
    df = args.df
    for dataset in args.datasets:
        data_x, data_y = load_dataset(args.data_dir, dataset)
        dim = int(data_x.shape[1])
        defaults = DATASET_DEFAULTS[dataset]
        idx = 0
        for lam in (args.priors or [0.1, 0.3]):
            eta = args.eta if args.eta is not None else min(0.03 / (lam + 0.25), 0.08)
            s2 = (df - 2.0) / (df * lam)  # t variance df/(df-2)*s2 == 1/lam (Gaussian prior variance)
            design = {
                "model": "tprior",
                "dataset": dataset,
                "dim": dim,
                "chains": args.chains,
                "prior": lam,
                "df": df,
                "t_scale_sq": s2,
                "eta": eta,
                "steps": args.steps or defaults["steps"],
                "burnin": args.burnin or defaults["burnin"],
                "reps": args.reps or defaults["reps"],
                "replicate": args.replicate,
                "seed": TPRIOR_SEED_BASE[dataset] + 10000 * args.replicate + idx,
                "design_seed": TPRIOR_SEED_BASE[dataset] + idx,
            }
            idx += 1
            grad_fn = lambda s: tprior_grad(s, data_x, data_y, df, s2, args.chunk_rows)
            for method in args.methods:
                result = run_method(design, grad_fn, method, beta_dim=dim)
                out_file.write(json.dumps(result, sort_keys=True) + "\n")
                out_file.flush()
                print(json.dumps(result, sort_keys=True), flush=True)


def run_hierarchical(args, out_file):
    data_x, data_y, gmat, n_groups = load_hierarchical(args.data_dir, args.raw_dir)
    if args.selftest:
        ok = selftest_hier(data_x, data_y, gmat, args.lam_beta)
        raise SystemExit(0 if ok else 1)
    beta_dim = int(data_x.shape[1])
    dim = beta_dim + n_groups + 1
    eta0 = args.eta if args.eta is not None else min(0.03 / (0.1 + 0.25), 0.08)
    design = {
        "model": "hierarchical",
        "dataset": "adult",
        "dim": dim,
        "chains": args.chains,
        "prior": args.lam_beta,
        "n_groups": n_groups,
        "eta": eta0,
        "eta0": eta0,
        "eta_halvings": 0,
        "calibrated": False,
        "calibrate_ok": True,
        "steps": args.steps or 700,
        "burnin": args.burnin or 350,
        "reps": args.reps or 64,
        "replicate": args.replicate,
        "seed": HIER_SEED_BASE + 10000 * args.replicate,
        "design_seed": HIER_SEED_BASE,
    }
    grad_fn = lambda s: hier_grad(s, data_x, data_y, gmat, args.lam_beta, args.chunk_rows)
    if args.calibrate:
        eta, halvings, ok = calibrate_eta(design, grad_fn)
        design.update({"eta": eta, "eta_halvings": halvings, "calibrated": True, "calibrate_ok": ok})
        print(json.dumps({"calibrate": {"eta0": eta0, "eta": eta, "halvings": halvings, "ok": ok}},
                         sort_keys=True), flush=True)
    for method in args.methods:
        result = run_method(design, grad_fn, method, beta_dim=beta_dim)
        out_file.write(json.dumps(result, sort_keys=True) + "\n")
        out_file.flush()
        print(json.dumps(result, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tprior", "hierarchical"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=["wdbc", "spambase", "adult"],
                        default=["wdbc", "spambase", "adult"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["iid", "zero_sum", "antithetic"])
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--priors", nargs="+", type=float, default=None)
    parser.add_argument("--df", type=float, default=3.0)
    parser.add_argument("--lam-beta", type=float, default=0.1)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.priors = args.priors or [0.3]
        args.reps = args.reps or 16
        args.steps = args.steps or 40
        args.burnin = args.burnin or 20
    with args.out.open("a", encoding="utf-8") as f:
        if args.model == "tprior":
            run_tprior(args, f)
        else:
            run_hierarchical(args, f)


if __name__ == "__main__":
    main()
