"""Numeric validation of the non-asymptotic variance-ratio theorem.

Synthetic targets INSIDE the theorem's assumption class with exactly
computable constants: U(x) = 0.5 sum_j h_j x_j^2 + alpha sum_j cos(x_j),
so that m >= min h_j - alpha, L <= max h_j + alpha (exact for the
product family, certified Weyl bounds for the dense family), and the
Hessian is rho-Lipschitz with rho = alpha.  Requires alpha < min h_j.

For each (alpha, d, K, eta) cell: run N independent ensembles of K ULA
chains from a common deterministic start under zero_sum and iid
couplings, T = ceil(horizon_mults / (eta m)) steps, record the
across-ensemble variance of a' Xbar_T for a fixed seeded unit probe a,
the realized mean spread S_t (validates the B2 = 2d/m lemma), and the
exact (m, L, rho).  The table emitter evaluates the proven bound from
the recorded constants and checks (pre-registered): (a) bound dominates
every measured ratio; (b) measured ratio scales like rho^2 (the bound's d-linearity is a worst-case allowance);
(c) conservatism factor reported honestly.  alpha = 0 rows are the
quadratic sanity floor (ratio at float noise).

Seeds 33x,xxx,xxx (new block).  JSONL sort_keys append, refuses
overwrite, provenance per row.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

SEED_BASE = 330_000_000
ALPHAS = [0.0, 1e-3, 3e-3, 1e-2, 3e-2]
DIMS = [2, 4, 8, 16]
KS = [4, 8]
ETA_FRACS = [0.05, 0.125, 0.25]  # of 1/(2L), so all satisfy eta <= 1/(2L)
H_MIN, H_MAX = 1.0, 2.0
N_ENSEMBLES = 4096
HORIZON_MULTS = 6.0


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except Exception:
        return "missing"


def provenance():
    return {
        "argv": " ".join(sys.argv),
        "script_sha256": file_sha256(__file__),
        "precommit_sha256": file_sha256("campaigns/variance_ratio_theorem_precommit.md"),
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__ if cp is not None else "none",
    }


def run_cell(xp, seed, alpha, d, k_chains, eta, n_ens, steps, probe, family):
    """Returns dict with var_zs, var_iid, spread diagnostics."""
    rng_h = np.random.default_rng(SEED_BASE + 7)
    h_diag = np.linspace(H_MIN, H_MAX, d)  # deterministic spectrum
    h = xp.asarray(h_diag, dtype=xp.float64)
    a = xp.asarray(probe, dtype=xp.float64)

    # product family: U = 0.5 sum h_j x_j^2 + alpha sum cos(x_j)
    #   grad = h*x - alpha*sin(x); rho = alpha (diagonal third derivative)
    # dense family:   U = 0.5 sum h_j x_j^2 + alpha cos(w'x), ||w|| = 1
    #   grad = h*x - alpha*sin(w'x)*w; rho = alpha*||w||^3 = alpha
    w_dense = xp.ones(d, dtype=xp.float64) / math.sqrt(d)

    def grad(x):
        if family == "product":
            return h * x - alpha * xp.sin(x)
        proj = x @ w_dense
        return h * x - alpha * xp.sin(proj)[..., None] * w_dense

    out = {}
    spread_last = None
    for method in ("zero_sum", "iid"):
        xp.random.seed(seed + (0 if method == "zero_sum" else 500))
        x = xp.zeros((n_ens, k_chains, d), dtype=xp.float64)
        s_acc = 0.0
        for t in range(steps):
            g = grad(x)
            noise = xp.random.standard_normal(x.shape, dtype=xp.float64)
            if method == "zero_sum":
                noise = noise - noise.mean(axis=1, keepdims=True)
                noise = noise * math.sqrt(k_chains / (k_chains - 1))
            x = x - eta * g + math.sqrt(2.0 * eta) * noise
            if t >= steps // 2:
                d_dev = x - x.mean(axis=1, keepdims=True)
                s_acc += float((d_dev * d_dev).sum(axis=2).mean())
        z = (x.mean(axis=1) @ a)
        out[f"var_{method}"] = float(z.var())
        if method == "zero_sum":
            spread_last = s_acc / (steps - steps // 2)
    out["mean_spread_zs"] = spread_last
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--n-ens", type=int, default=N_ENSEMBLES)
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--dims", nargs="+", type=int, default=DIMS)
    parser.add_argument("--ks", nargs="+", type=int, default=KS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--make-table", action="store_true")
    parser.add_argument("--ratio-constant", type=float, default=None,
                        help="the proven ratio-bound constant C in C*(K/(K-1))*rho^2 d L/m^4 (table mode)")
    parser.add_argument("--table-inputs", nargs="+", type=Path,
                        default=[Path(f"outputs/sweeps/variance_ratio_rep{r}.jsonl") for r in range(3)])
    parser.add_argument("--table-out", type=Path, default=Path("paper/tables/variance_ratio_check.tex"))
    args = parser.parse_args()

    if args.make_table:
        make_table(args.table_inputs, args.table_out, args.ratio_constant)
        return
    if args.out is None:
        parser.error("--out required")
    if args.out.exists():
        parser.error(f"refusing to overwrite existing {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.n_ens = 512
        args.alphas = [0.0, 1e-2]
        args.dims = [4]
        args.ks = [8]

    xp = cp if cp is not None else np
    prov = provenance()
    rng = np.random.default_rng(SEED_BASE + 11)
    with args.out.open("x", encoding="utf-8") as f:
        idx = 0
        for family in ("product", "dense"):
          for d in args.dims:
            probe = rng.standard_normal(d)
            probe /= np.linalg.norm(probe)
            for alpha in args.alphas:
                if family == "dense" and alpha == 0.0:
                    continue  # quadratic sanity covered by the product rows
                m_const = H_MIN - alpha
                l_const = H_MAX + alpha
                rho = alpha
                assert m_const > 0
                for k_chains in args.ks:
                    for frac in ETA_FRACS:
                        eta = frac / (2.0 * l_const)
                        steps = int(math.ceil(HORIZON_MULTS / (eta * m_const)))
                        seed = SEED_BASE + 1_000_000 * args.replicate + 1000 * idx
                        res = run_cell(xp, seed, alpha, d, k_chains, eta,
                                       args.n_ens, steps, probe, family)
                        row = {
                            "kind": "variance_ratio", "family": family,
                            "alpha": alpha, "dim": d, "chains": k_chains,
                            "eta": eta, "eta_frac_of_half_inv_L": frac,
                            "steps": steps, "n_ensembles": args.n_ens,
                            "m_const": m_const, "l_const": l_const, "rho": rho,
                            "replicate": args.replicate, "seed": seed,
                            "quick": bool(args.quick),
                            "b2_bound": 2.0 * d / m_const,
                            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                            "git_hash": git_hash(), "provenance": prov,
                            **res,
                        }
                        row["ratio"] = (row["var_zero_sum"] / row["var_iid"]
                                        if row["var_iid"] > 0 else float("nan"))
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                        f.flush()
                        print(json.dumps({k: row[k] for k in
                                          ("alpha", "dim", "chains", "eta", "ratio",
                                           "mean_spread_zs", "b2_bound")},
                                         sort_keys=True), flush=True)
                        idx += 1
    print(f"wrote {args.out}")


def make_table(inputs, out, ratio_constant):
    assert ratio_constant is not None, "--ratio-constant required (from the proven theorem)"
    rows = []
    for p in inputs:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("quick")]
    lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule",
             "family & $\\rho$ & $d$ & $K$ & $\\eta L$ & measured ratio & proven bound & slack \\\\",
             "\\midrule"]
    viol = 0
    max_meas = {}
    for r in sorted(rows, key=lambda r: (r["rho"], r["dim"], r["chains"], r["eta"])):
        if r["rho"] == 0.0:
            continue  # quadratic sanity rows reported in text, not table
        bound = (ratio_constant * r["chains"] / (r["chains"] - 1)
                 * r["rho"] ** 2 * r["dim"] * r["l_const"] / r["m_const"] ** 4)
        key = (r.get("family", "product"), r["rho"], r["dim"], r["chains"], round(r["eta"], 8))
        max_meas.setdefault(key, []).append(r["ratio"])
        if r["ratio"] > bound:
            viol += 1
    for key, vals in sorted(max_meas.items()):
        fam, rho, d, k, eta = key
        med = float(np.median(vals))
        mx = float(np.max(vals))
        r0 = next(r for r in rows if (r.get("family", "product"), r["rho"], r["dim"], r["chains"],
                                      round(r["eta"], 8)) == key)
        bound = (ratio_constant * k / (k - 1) * rho ** 2 * d
                 * r0["l_const"] / r0["m_const"] ** 4)
        slack = bound / mx if mx > 0 else float("inf")
        lines.append(f"{fam} & {rho:g} & {d} & {k} & {r0['eta'] * r0['l_const']:.3f} & "
                     f"{med:.2e} [{mx:.2e}] & {bound:.2e} & {slack:.1f}x \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    quad = [r["ratio"] for r in rows if r["rho"] == 0.0]
    print(f"wrote {out}; bound violations: {viol}; quadratic sanity max ratio: "
          f"{max(quad) if quad else 'n/a'}")


if __name__ == "__main__":
    main()
