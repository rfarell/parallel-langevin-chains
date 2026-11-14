"""Certified-Gamma screen (screen S1, 2026-07-17).

Pre-registered screen (protocol section "Screen S1").  It decides whether a
local-Gamma version of the variance-ratio theorem is worth pursuing on the
real-data cells; it is a deterministic computation, not a sampling result.

For each of the six real cells (wdbc/spambase/adult x lambda in {0.1, 0.3}),
tempered convention shared with mala_real_logistic.py / rr_extrapolation.py /
map_baseline.py:

    U(theta) = (1/n) sum_i [softplus(x_i . theta) - y_i (x_i . theta)]
               + (lambda/2) ||theta * mask||^2,   mask[0] = 0
               (coordinate 0 is the prepended intercept column, UNPENALIZED),

data matrix = float32 feature cast + intercept column, promoted to float64.

At BOTH evaluation points theta in {theta_MAP (full-Hessian Newton, float64,
||grad|| <= 1e-10), archived gold posterior mean}:

    H(theta) = (1/n) X' D X + lambda P,  D_ii = sigma_i (1 - sigma_i),
               P = diag(mask)  (identity with 0 on the intercept)
    m_ub   = lambda_min(H)   (upper bound on any ball-infimum m)
    L_lb   = lambda_max(H)   (lower bound on any ball-supremum L)
    rho_lb = max over the d eigenvector directions u of H of
             |D^3 U(theta)[u,u,u]|,
             D^3 U[u,u,u] = (1/n) sum_i sigma_i (1-sigma_i)(1-2 sigma_i) (x_i'u)^3
             (exact directional third derivative; the prior is quadratic and
             contributes nothing; a valid lower bound on the ball rho)
    Gamma_lb = rho_lb^2 * d * L_lb / m_ub^4,   ratio = Gamma_lb / (1/32).

Certified LOWER bound on the local Gamma over any ball containing the point.

Pre-registered decision rule: every cell ratio >= 1e3 -> the local-Gamma
theorem is ruled out (table goes to the appendix); any cell ratio < 10 ->
reopen it for that cell.

Self-checks: theta_MAP vs outputs/sweeps/map_baseline.jsonl (if present); dims
vs the paper's (wdbc 31, spambase 58, adult 105) and vs the gold rows.

Writes outputs/screens/certified_gamma_screen.jsonl (refuses to overwrite):
one object per cell x evaluation point, with provenance (argv, script sha256,
numpy version, data + gold file sha256s, git hash).  Deterministic -- no seeds.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

DATASETS = ["wdbc", "spambase", "adult"]
PRIORS = [0.1, 0.3]
EXPECTED_DIM = {"wdbc": 31, "spambase": 58, "adult": 105}
THRESHOLD = 1.0 / 32.0


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
            return theta, gnorm, it
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


def certified_quantities(theta, x, y, prior, mask):
    """Exact at-a-point m_ub, L_lb, rho_lb per the registered convention."""
    n, d = x.shape
    z = x @ theta
    p = sigmoid_stable(z)
    w = p * (1.0 - p)
    hess = (x.T * w) @ x / n + prior * np.diag(mask)
    evals, evecs = np.linalg.eigh(hess)
    m_ub = float(evals[0])
    L_lb = float(evals[-1])
    # Directional third derivatives over all d (orthonormal) eigenvector directions.
    w3 = w * (1.0 - 2.0 * p)               # sigma(1-sigma)(1-2 sigma) per point
    t = x @ evecs                          # n x d: t[:, j] = x_i . u_j
    d3 = (w3 @ t**3) / n                   # D^3 U[u_j, u_j, u_j] for each j
    j = int(np.argmax(np.abs(d3)))
    rho_lb = float(np.abs(d3[j]))
    return {
        "m_ub": m_ub,
        "L_lb": L_lb,
        "rho_lb": rho_lb,
        "rho_argmax_eigindex": j,          # 0 = smallest eigenvalue of H
        "rho_argmax_eigval": float(evals[j]),
        "d3_min_eigdir": float(d3[0]),
        "d3_max_eigdir": float(d3[-1]),
    }


def load_gold(path):
    gold = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "gold":
            gold[(row["dataset"], row["prior"])] = row
    return gold


def load_map_baseline(path):
    if not path.exists():
        return None
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "map_baseline":
            rows[(r["dataset"], r["prior"])] = r
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--gold", type=Path, default=Path("outputs/sweeps/rr_extrapolation_gold.jsonl"))
    parser.add_argument("--map-baseline", type=Path, default=Path("outputs/sweeps/map_baseline.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("outputs/screens/certified_gamma_screen.jsonl"))
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    provenance = {
        "argv": sys.argv,
        "script_sha256": sha256_file(Path(__file__)),
        "numpy_version": np.__version__,
        "gold_sha256": sha256_file(args.gold),
        "git_hash": git_hash(),
    }

    gold = load_gold(args.gold)
    map_base = load_map_baseline(args.map_baseline)
    rows = []
    selfcheck_lines = []
    for ds in DATASETS:
        x, y = load_dataset(args.data_dir, ds)
        n, d = x.shape
        assert d == EXPECTED_DIM[ds], f"{ds}: dim {d} != paper's {EXPECTED_DIM[ds]}"
        mask = np.ones(d)
        mask[0] = 0.0
        data_sha = sha256_file(args.data_dir / f"{ds}.npz")
        for prior in PRIORS:
            g = gold[(ds, prior)]
            assert d == g["dim"], (ds, prior, d, g["dim"])
            theta_map, gnorm, iters = newton_map(x, y, prior, tol=args.tol, max_iter=args.max_iter)
            gold_mean = np.asarray(g["gold_mean"], dtype=np.float64)
            # Self-check vs the accepted MAP artifact (same convention).
            map_diff = None
            if map_base is not None and (ds, prior) in map_base:
                ref = np.asarray(map_base[(ds, prior)]["theta_map"], dtype=np.float64)
                map_diff = float(np.max(np.abs(theta_map - ref)))
                selfcheck_lines.append(
                    f"  {ds} prior={prior}: max|theta_MAP - map_baseline| = {map_diff:.3e}")
            for point_name, theta in [("map", theta_map), ("gold_mean", gold_mean)]:
                q = certified_quantities(theta, x, y, prior, mask)
                gamma_lb = q["rho_lb"] ** 2 * d * q["L_lb"] / q["m_ub"] ** 4
                row = {
                    "kind": "screen",
                    "screen": "S1_certified_gamma",
                    "precommit": "campaigns/screens_precommit.md#screen-s1",
                    "dataset": ds,
                    "prior": prior,
                    "point": point_name,
                    "dim": d,
                    "n_train": n,
                    "intercept_coord": 0,
                    **q,
                    "gamma_lb": gamma_lb,
                    "threshold": THRESHOLD,
                    "ratio_to_threshold": gamma_lb / THRESHOLD,
                    "gold_id": g["gold_id"],
                    "newton_iters": iters,
                    "newton_grad_norm": gnorm,
                    "map_vs_baseline_max_absdiff": map_diff,
                    "dist_map_to_gold_mean": float(np.linalg.norm(theta_map - gold_mean)),
                    "data_sha256": data_sha,
                    **provenance,
                }
                rows.append(row)

    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {args.out}  (screen artifact, {len(rows)} rows)\n")

    print("Self-checks:")
    print(f"  dims match paper: " + ", ".join(f"{ds}={EXPECTED_DIM[ds]}" for ds in DATASETS))
    if selfcheck_lines:
        print("\n".join(selfcheck_lines))
    else:
        print("  (no map_baseline artifact found; Newton MAP not cross-checked)")

    hdr = (f"\n{'dataset':9s} {'prior':5s} {'point':9s} {'m_ub':>10s} {'L_lb':>10s} "
           f"{'rho_lb':>10s} {'Gamma_lb':>10s} {'x(1/32)':>10s}")
    print(hdr)
    for r in rows:
        print(f"{r['dataset']:9s} {r['prior']:<5g} {r['point']:9s} {r['m_ub']:10.4g} "
              f"{r['L_lb']:10.4g} {r['rho_lb']:10.4g} {r['gamma_lb']:10.4g} "
              f"{r['ratio_to_threshold']:10.4g}")

    # Pre-registered decision rule (min over the two evaluation points per cell).
    per_cell = {}
    for r in rows:
        key = (r["dataset"], r["prior"])
        per_cell[key] = min(per_cell.get(key, float("inf")), r["ratio_to_threshold"])
    worst_cell = min(per_cell, key=per_cell.get)
    worst = per_cell[worst_cell]
    print(f"\nworst cell (min over both points): {worst_cell} ratio = {worst:.4g}")
    if worst >= 1e3:
        print("decision rule: ALL cells >= 1e3 x (1/32)  ->  local-Gamma theorem ruled out;"
              "\n               table goes to the appendix.")
    elif any(v < 10 for v in per_cell.values()):
        reopen = [k for k, v in per_cell.items() if v < 10]
        print(f"decision rule: cells below 10 x (1/32): {reopen}  ->  reopen the local-Gamma theorem for these cells.")
    else:
        print("decision rule: no cell < 10 x threshold, but not all >= 1e3 x threshold"
              " -> neither trigger fires; report as-is.")


if __name__ == "__main__":
    main()
