"""MAP (Newton) baseline for the exact-gold end-task comparison.

Baseline check: on these log-concave logistic posteriors, is the MAP (= mean of
the Laplace approximation) already a good posterior-mean estimate, making
sampling unnecessary?  Uses the same potential as scripts/mala_real_logistic.py
and scripts/rr_extrapolation.py:

    U(theta) = mean_i [softplus(x_i . theta) - y_i (x_i . theta)]
               + (prior/2) ||theta * mask||^2,   mask[0] = 0 (intercept free),

with the same data pipeline (intercept column prepended, features cast to
float32); Newton runs in float64.  Deterministic — no seeds.

Per cell (dataset x prior) reports ||theta_MAP - theta_gold||^2 against the
pooled gold posterior mean, the gold across-seed spread, the ratio to the
zero-sum ULA@4eta_c single-level median MSE, and trace of the Laplace
covariance H(theta_MAP)^{-1}.  Writes outputs/sweeps/map_baseline.jsonl
(refuses to overwrite) and prints the '{:.3g}'-formatted MAP column for
paper/tables/gold_endtask.tex (this script does not edit the table).
"""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

DATASETS = ["wdbc", "spambase", "adult"]
PRIORS = [0.1, 0.3]
DATASET_LABEL = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def add_intercept(x_np):
    # Same as the samplers: float32 feature cast defines the shared data matrix.
    ones = np.ones((x_np.shape[0], 1), dtype=np.float32)
    return np.concatenate([ones, x_np.astype(np.float32)], axis=1)


def load_dataset(data_dir, name):
    data = np.load(data_dir / f"{name}.npz", allow_pickle=True)
    x = add_intercept(data["X_train"]).astype(np.float64)
    y = data["y_train"].astype(np.float32).astype(np.float64)
    return x, y


def sigmoid_stable(z):
    p = np.empty_like(z)
    pos = z >= 0
    p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    p[~pos] = ez / (1.0 + ez)
    return p


def potential(theta, x, y, prior, mask):
    z = x @ theta
    u = np.mean(np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - y * z)
    return u + 0.5 * prior * np.sum((theta * mask) ** 2)


def grad_hess(theta, x, y, prior, mask):
    n = x.shape[0]
    z = x @ theta
    p = sigmoid_stable(z)
    grad = x.T @ (p - y) / n + prior * theta * mask
    w = p * (1.0 - p)
    hess = (x.T * w) @ x / n + prior * np.diag(mask)
    return grad, hess


def newton_map(x, y, prior, tol=1e-10, max_iter=200):
    d = x.shape[1]
    mask = np.ones(d)
    mask[0] = 0.0
    theta = np.zeros(d)
    u = potential(theta, x, y, prior, mask)
    for it in range(max_iter):
        grad, hess = grad_hess(theta, x, y, prior, mask)
        gnorm = float(np.linalg.norm(grad))
        if gnorm < tol:
            return theta, gnorm, it, hess
        step = cho_solve(cho_factor(hess, lower=True), grad)
        # Backtracking safeguard; full Newton steps are expected to pass.
        t = 1.0
        while t > 1e-8:
            cand = theta - t * step
            u_cand = potential(cand, x, y, prior, mask)
            if u_cand <= u - 1e-4 * t * float(grad @ step):
                theta, u = cand, u_cand
                break
            t *= 0.5
        else:
            raise RuntimeError(f"line search failed at iter {it} (gnorm={gnorm:.3e})")
    raise RuntimeError(f"Newton did not reach grad tol {tol} in {max_iter} iters (gnorm={gnorm:.3e})")


def load_gold(path):
    gold = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "gold":
            gold[(row["dataset"], row["prior"])] = row
    return gold


def load_zs_medians(paths):
    per_cell = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("kind") == "estimator" and r.get("estimator") == "single"
                    and r.get("coupling") == "zero_sum" and r.get("level") == "eta1"
                    and r.get("pair") == "primary"):
                per_cell.setdefault((r["dataset"], r["prior"]), []).append(r["mse_gold_median"])
    return per_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--gold", type=Path, default=Path("outputs/sweeps/rr_extrapolation_gold.jsonl"))
    parser.add_argument("--rr", nargs="+", type=Path,
                        default=[Path(f"outputs/sweeps/rr_extrapolation_rep{r}.jsonl") for r in range(3)])
    parser.add_argument("--out", type=Path, default=Path("outputs/sweeps/map_baseline.jsonl"))
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    gold = load_gold(args.gold)
    zs = load_zs_medians(args.rr)
    rows = []
    for ds in DATASETS:
        x, y = load_dataset(args.data_dir, ds)
        for prior in PRIORS:
            g = gold[(ds, prior)]
            assert x.shape[1] == g["dim"], (ds, prior, x.shape, g["dim"])
            theta, gnorm, iters, hess = newton_map(x, y, prior, tol=args.tol, max_iter=args.max_iter)
            gold_mean = np.asarray(g["gold_mean"], dtype=np.float64)
            map_err = float(np.sum((theta - gold_mean) ** 2))
            sigma = cho_solve(cho_factor(hess, lower=True), np.eye(hess.shape[0]))
            trace_sigma = float(np.trace(sigma))
            zs_reps = zs[(ds, prior)]
            zs_med = float(statistics.median(zs_reps))
            rows.append({
                "kind": "map_baseline",
                "target": "rr_extrapolation",
                "dataset": ds,
                "prior": prior,
                "dim": int(x.shape[1]),
                "n_train": int(x.shape[0]),
                "gold_id": g["gold_id"],
                "newton_iters": iters,
                "grad_norm": gnorm,
                "map_err_sq_gold": map_err,
                "gold_spread": g["gold_spread"],
                "zs_ula_eta1_median": zs_med,
                "zs_ula_eta1_reps": zs_reps,
                "map_over_zs_ratio": map_err / zs_med,
                "trace_sigma_laplace": trace_sigma,
                "theta_map": [float(v) for v in theta],
                "git_hash": git_hash(),
            })

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {args.out}\n")

    hdr = f"{'dataset':9s} {'prior':5s} {'iters':5s} {'gradnorm':9s} {'||MAP-gold||^2':>14s} {'gold_spread':>11s} {'zs_ULA@e1':>10s} {'MAP/zs':>9s} {'tr(Sigma)':>10s}"
    print(hdr)
    for r in rows:
        print(f"{r['dataset']:9s} {r['prior']:<5g} {r['newton_iters']:<5d} {r['grad_norm']:.1e} "
              f"{r['map_err_sq_gold']:14.6g} {r['gold_spread']:11.2e} {r['zs_ula_eta1_median']:10.4g} "
              f"{r['map_over_zs_ratio']:9.3g} {r['trace_sigma_laplace']:10.4g}")

    print("\nMAP column for paper/tables/gold_endtask.tex ('{:.3g}' style, row order as table):")
    for r in rows_table_order(rows):
        print(f"{DATASET_LABEL[r['dataset']]} & {r['prior']}  ->  MAP cell: {r['map_err_sq_gold']:.3g}")


def rows_table_order(rows):
    idx = {(r["dataset"], r["prior"]): r for r in rows}
    return [idx[(ds, pr)] for ds in DATASETS for pr in PRIORS]


if __name__ == "__main__":
    main()
