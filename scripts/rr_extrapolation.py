"""Richardson-Romberg (RR) extrapolation with coupled Langevin ensembles.

Implements the pre-committed protocol in
campaigns/rr_extrapolation_precommit.md.

Idea: ULA's posterior-MEAN weak error is O(eta) on nonquadratic targets.
theta_RR = 2*theta(eta/2) - theta(eta) cancels the O(eta) term; zero-sum
coupling makes each level's ensemble-mean noise ~0, so the extrapolation
differencing does not pay the classical variance penalty.  All estimators
are scored against an unbiased MALA gold standard (acceptance ~0.5-0.7,
K=64 chains, 5000 burn-in + 30000 kept steps, 3 independent seeds, pooled).

Conventions follow scripts/real_logistic_gpu.py and
scripts/mala_real_logistic.py (data loading, logistic U/grad, make_noise
couplings, MALA accept step, JSONL append with sort_keys+flush, git_hash,
--quick, warm-up-excluded timing).  Accumulators here are float64 (30k-step
gold sums drift at fp32); dynamics stay float32 like the parents.

Protocol P4 additions (campaigns/screens_precommit.md,
"Protocol P4" + Amendment v2 section E + Amendment v3 V3-1/V3-3/V3-4/V3-5 +
Amendment v4 V4-2 diagnostics and V4-4 attempt-specific init streams):
--gold-only recomputes ONLY the gold references with frozen archived etas,
registered start laws (zero / chol-dispersed / gauss) and registered seed
streams, plus batch-mean / split-half / post-burn-acceptance bookkeeping,
per-coordinate rank-normalized split-Rhat (bulk+folded, gated on max) and
bulk ESS over the per-step ensemble-mean series (series archived to an .npz
sidecar per cell/arm/attempt); --selftest-p4 runs the CPU-only unit checks.
Accepted P4 runs remain gated (V3-7).
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

cp = None  # imported lazily; --selftest is CPU/numpy only

DATASETS = ["wdbc", "spambase", "adult"]
DATASET_CFG = {
    "wdbc": {"n_ens": 128, "budget": 2400},
    "spambase": {"n_ens": 128, "budget": 2400},
    "adult": {"n_ens": 64, "budget": 1800},
}
GOLD_CHAINS = 64
GOLD_BURNIN = 5000
GOLD_KEPT = 30000
GOLD_N_SEEDS = 3
PILOT_CHAINS = 32
PILOT_STEPS = 300  # acceptance counted over the last 2/3 (zero-start transient discarded)
# MALA acceptance at 16x base eta is still ~0.85 on these cells (see
# outputs/sweeps/mala_real_rep0.jsonl), so the scan extends far above 16x
# to reach the 0.5-0.7 gold band.  Descending: pick the largest step in band.
PILOT_SCAN = [96.0, 80.0, 64.0, 48.0, 40.0, 32.0, 24.0, 20.0, 16.0, 12.0, 8.0, 6.0, 4.0, 2.0, 1.0]
# Seed bases: 11xxxxxx, disjoint from all prior campaigns (6xxxxx/7xxxxx).
SEED_BASE = 11000000
GOLD_SEED_BASE = 11900000
PILOT_SEED_BASE = 11950000
DS_OFF = {"wdbc": 0, "spambase": 100000, "adult": 200000}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def default_eta(prior):
    # Campaign heuristic eta_c.
    return min(0.03 / (prior + 0.25), 0.08)


# ----------------------------------------------------------------------------
# Selftest (CPU, numpy): 1D asymmetric strongly-convex target.
#
# U(x) = x^2/2 + c*x^3/(1+x^2), c = 0.3.
#   U'(x)  = x + c*x^2*(3+x^2)/(1+x^2)^2
#   U''(x) = 1 + c*(6x-2x^3)/(1+x^2)^3
# max_x |(6x-2x^3)/(1+x^2)^3| = 1.4563 (at x^2 = 3-2*sqrt(2)), so
# U'' >= 1 - 0.3*1.4563 = 0.5631 > 0 everywhere: globally strongly convex,
# m ~ 0.563, L ~ 1.437 (ULA stable for eta << 2/L ~ 1.39).
# U'''(0) = 6c = 1.8 != 0: nonzero third derivative at the mode, so the
# stationary ULA mean carries a genuine O(eta) shift.
# ----------------------------------------------------------------------------

ST_C = 0.3


def st_u(x):
    return 0.5 * x * x + ST_C * x ** 3 / (1.0 + x * x)


def st_grad(x):
    x2 = x * x
    return x + ST_C * x2 * (3.0 + x2) / (1.0 + x2) ** 2


def st_upp(x):
    x2 = x * x
    return 1.0 + ST_C * (6.0 * x - 2.0 * x * x2) / (1.0 + x2) ** 3


def st_true_mean():
    xs = np.linspace(-12.0, 12.0, 480001)
    dx = xs[1] - xs[0]
    w = np.exp(-(st_u(xs) - st_u(xs).min()))
    trap = lambda f: float(np.sum((f[1:] + f[:-1])) * 0.5 * dx)
    return trap(xs * w) / trap(w)


def st_level(rng, eta, steps, n_ens, chains, coupling):
    """Per-ensemble time-average (50% burn-in) of the ensemble mean."""
    x = np.zeros((n_ens, chains))
    burn = steps // 2
    acc = np.zeros(n_ens)
    windows = 0
    sqrt_2eta = math.sqrt(2.0 * eta)
    scale = math.sqrt(chains / (chains - 1.0))
    for step in range(steps):
        z = rng.standard_normal((n_ens, chains))
        if coupling == "zero_sum":
            z = (z - z.mean(axis=1, keepdims=True)) * scale
        x = x - eta * st_grad(x) + sqrt_2eta * z
        if step >= burn:
            acc += x.mean(axis=1)
            windows += 1
    return acc / windows


def selftest():
    rng = np.random.default_rng(20260713)
    chains = 8
    # Bias fits need tight SEs (RR biases are O(eta^2), tiny): 1024 ensembles
    # x 9600 time units.  ULA stable well past eta=0.8 (2/L ~ 1.39).
    n_ens, t_phys = 1024, 9600.0
    etas = [0.8, 0.4, 0.2, 0.1]
    true_mean = st_true_mean()
    grid = np.linspace(-8.0, 8.0, 20001)
    min_upp = float(st_upp(grid).min())
    print(f"selftest target: U(x)=x^2/2+{ST_C}*x^3/(1+x^2)  true_mean={true_mean:+.6f}  "
          f"min U'' on [-8,8]={min_upp:.4f} (analytic 0.5631)  U'''(0)={6*ST_C:.2f}")
    ok_convex = min_upp > 0.5

    bias, se = {}, {}
    for eta in etas:
        steps = int(round(t_phys / eta))
        avg = st_level(rng, eta, steps, n_ens, chains, "zero_sum")
        b = float(avg.mean()) - true_mean
        s = float(avg.std(ddof=1)) / math.sqrt(n_ens)
        bias[eta], se[eta] = b, s
        print(f"  ULA  eta={eta:5.3f} steps={steps:6d}  bias={b:+.3e}  se={s:.1e}")

    # (i) ULA mean bias ~ O(eta): slope of log|bias| vs log eta.
    ula_pts = [(e, abs(bias[e])) for e in etas if abs(bias[e]) > 4 * se[e]]
    ula_slope = float(np.polyfit(np.log([p[0] for p in ula_pts]), np.log([p[1] for p in ula_pts]), 1)[0]) if len(ula_pts) >= 2 else float("nan")
    ok_ula_pts = len(ula_pts) == len(etas)

    # (ii) RR bias ~ O(eta^2): theta_RR = 2*theta(eta/2) - theta(eta) so
    # bias_RR(eta) = 2*bias(eta/2) - bias(eta) exactly (linearity).
    rr_pts = []
    for e1, e2 in zip(etas[:-1], etas[1:]):
        b_rr = 2 * bias[e2] - bias[e1]
        s_rr = math.sqrt(4 * se[e2] ** 2 + se[e1] ** 2)
        used = abs(b_rr) > 4 * s_rr
        print(f"  RR   pair=({e1},{e2})  bias={b_rr:+.3e}  se={s_rr:.1e}  used_in_fit={used}")
        if used:
            rr_pts.append((e1, abs(b_rr)))
    rr_slope = float(np.polyfit(np.log([p[0] for p in rr_pts]), np.log([p[1] for p in rr_pts]), 1)[0]) if len(rr_pts) >= 2 else float("nan")
    ok_rr_pts = len(rr_pts) >= 2

    # (iii) zero-sum vs iid across-ensemble variance of each level's
    # time-average (K=8; lighter run suffices for an orders-of-magnitude call).
    ratios = {}
    for eta in [0.4, 0.2]:
        steps = int(round(3200.0 / eta))
        v_iid = float(st_level(rng, eta, steps, 512, chains, "iid").var(ddof=1))
        v_zs = float(st_level(rng, eta, steps, 512, chains, "zero_sum").var(ddof=1))
        ratios[eta] = v_iid / v_zs
        print(f"  VAR  eta={eta:5.3f}  iid={v_iid:.3e}  zero_sum={v_zs:.3e}  ratio={v_iid / v_zs:.1f}x")

    ok_ula = ok_ula_pts and 0.7 <= ula_slope <= 1.3
    ok_rr = ok_rr_pts and rr_slope >= 1.5
    ok_var = all(r >= 100.0 for r in ratios.values())
    print(f"  slopes: ULA={ula_slope:.3f} (PASS if in [0.7,1.3])  RR={rr_slope:.3f} (PASS if >=1.5)")
    print(f"  var ratios >=100x: {ok_var}  convexity>0.5: {ok_convex}")
    verdict = ok_convex and ok_ula and ok_rr and ok_var
    print(f"selftest {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


# ----------------------------------------------------------------------------
# GPU pieces (mirrors of the parent scripts).
# ----------------------------------------------------------------------------

def make_noise(shape, method, dtype=None):
    dtype = dtype or cp.float32
    reps, chains, _ = shape
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        z = cp.random.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
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


def mala_step(x, u_x, g_x, noise, eta, prior, data_x, data_y, chunk_rows, accept_acc):
    sqrt_2eta = math.sqrt(2.0 * eta)
    prop = x - eta * g_x + sqrt_2eta * noise
    u_p, g_p = potential_and_grad(prop, data_x, data_y, prior, chunk_rows)
    fwd = cp.sum((prop - x + eta * g_x) ** 2, axis=-1)
    rev = cp.sum((x - prop + eta * g_p) ** 2, axis=-1)
    log_alpha = u_x - u_p + (fwd - rev) / (4.0 * eta)
    accept = cp.log(cp.random.random(x.shape[:-1], dtype=cp.float32)) < log_alpha
    accept_acc += accept.mean()
    acc3 = accept[..., None]
    return cp.where(acc3, prop, x), cp.where(accept, u_p, u_x), cp.where(acc3, g_p, g_x)


def ula_time_avg(seed, coupling, eta, steps, n_ens, chains, dim, prior, data_x, data_y, chunk_rows):
    """One level: ULA from zero starts, 50% burn-in; returns per-ensemble
    float64 time-average of the K-chain ensemble mean, shape (n_ens, dim)."""
    cp.random.seed(seed)
    x = cp.zeros((n_ens, chains, dim), dtype=cp.float32)
    burn = steps // 2
    acc = cp.zeros((n_ens, dim), dtype=cp.float64)
    sqrt_2eta = math.sqrt(2.0 * eta)
    windows = 0
    for step in range(steps):
        grad = logistic_grad(x, data_x, data_y, prior, chunk_rows)
        x = x - eta * grad + sqrt_2eta * make_noise(x.shape, coupling)
        if step >= burn:
            acc += x.mean(axis=1, dtype=cp.float64)
            windows += 1
    return acc / windows


def mala_time_avg(seed, coupling, eta, steps, n_ens, chains, dim, prior, data_x, data_y, chunk_rows):
    cp.random.seed(seed)
    x = cp.zeros((n_ens, chains, dim), dtype=cp.float32)
    u, g = potential_and_grad(x, data_x, data_y, prior, chunk_rows)
    burn = steps // 2
    acc = cp.zeros((n_ens, dim), dtype=cp.float64)
    accept_acc = cp.zeros((), dtype=cp.float32)
    windows = 0
    for step in range(steps):
        x, u, g = mala_step(x, u, g, make_noise(x.shape, coupling), eta, prior, data_x, data_y, chunk_rows, accept_acc)
        if step >= burn:
            acc += x.mean(axis=1, dtype=cp.float64)
            windows += 1
    return acc / windows, float((accept_acc / steps).get())


def warm_up(kernel, coupling, eta, chains, dim, prior, data_x, data_y, chunk_rows):
    w = cp.zeros((2, chains, dim), dtype=cp.float32)
    if kernel == "mala":
        u, g = potential_and_grad(w, data_x, data_y, prior, chunk_rows)
        acc = cp.zeros((), dtype=cp.float32)
        mala_step(w, u, g, make_noise(w.shape, coupling), eta, prior, data_x, data_y, chunk_rows, acc)
    else:
        g = logistic_grad(w, data_x, data_y, prior, chunk_rows)
        w = w - eta * g + math.sqrt(2.0 * eta) * make_noise(w.shape, coupling)
    cp.cuda.Stream.null.synchronize()


# ----------------------------------------------------------------------------
# Gold standard: MALA, pilot-calibrated step (acceptance ~0.5-0.7), K=64,
# 5000 burn-in + 30000 kept, 3 seeds, pooled.  Cached in a gold JSONL and
# copied (kind=gold) into every replicate file; gold seeds are fixed and
# shared across replicates.
# ----------------------------------------------------------------------------

def calibrate_gold_eta(dataset, prior, dim, data_x, data_y, chunk_rows, pilot_steps):
    base = default_eta(prior)
    seed = PILOT_SEED_BASE + DS_OFF[dataset] + int(round(prior * 1000))
    rates = {}
    chosen = None
    pilot_burn = pilot_steps // 3  # acceptance from a zero start is transient-biased low
    for scale in PILOT_SCAN:
        eta = scale * base
        cp.random.seed(seed + int(round(scale * 1000)))
        x = cp.zeros((1, PILOT_CHAINS, dim), dtype=cp.float32)
        u, g = potential_and_grad(x, data_x, data_y, prior, chunk_rows)
        accept_acc = cp.zeros((), dtype=cp.float32)
        for step in range(pilot_steps):
            noise = cp.random.standard_normal(x.shape, dtype=cp.float32)
            x, u, g = mala_step(x, u, g, noise, eta, prior, data_x, data_y, chunk_rows, accept_acc)
            if step == pilot_burn - 1:
                accept_acc *= 0.0
        rate = float((accept_acc / (pilot_steps - pilot_burn)).get())
        rates[scale] = rate
        if chosen is None and 0.5 <= rate <= 0.7:
            chosen = (scale, eta, rate)
    if chosen is None:
        scale = min(rates, key=lambda s: abs(rates[s] - 0.6))
        chosen = (scale, scale * base, rates[scale])
    return chosen, rates


def gold_seed_run(seed, eta, burnin, kept, chains, dim, prior, data_x, data_y, chunk_rows):
    cp.random.seed(seed)
    x = cp.zeros((1, chains, dim), dtype=cp.float32)
    u, g = potential_and_grad(x, data_x, data_y, prior, chunk_rows)
    acc = cp.zeros(dim, dtype=cp.float64)
    accept_acc = cp.zeros((), dtype=cp.float32)
    for step in range(burnin + kept):
        noise = cp.random.standard_normal(x.shape, dtype=cp.float32)
        x, u, g = mala_step(x, u, g, noise, eta, prior, data_x, data_y, chunk_rows, accept_acc)
        if step >= burnin:
            acc += x.mean(axis=(0, 1), dtype=cp.float64)
    return acc / kept, float((accept_acc / (burnin + kept)).get())


def compute_gold(dataset, prior, dim, data_x, data_y, chunk_rows, gold_cfg):
    burnin, kept, chains, pilot_steps = gold_cfg["burnin"], gold_cfg["kept"], gold_cfg["chains"], gold_cfg["pilot_steps"]
    seeds = [GOLD_SEED_BASE + DS_OFF[dataset] + int(round(prior * 1000)) + j for j in range(GOLD_N_SEEDS)]
    gold_id = f"{dataset}|prior={prior}|chains={chains}|burnin={burnin}|kept={kept}|seeds={seeds[0]}-{seeds[-1]}"
    t0 = time.perf_counter()
    (scale, eta, pilot_rate), rates = calibrate_gold_eta(dataset, prior, dim, data_x, data_y, chunk_rows, pilot_steps)
    means, accepts = [], []
    for s in seeds:
        m, a = gold_seed_run(s, eta, burnin, kept, chains, dim, prior, data_x, data_y, chunk_rows)
        means.append(m)
        accepts.append(a)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    pooled = sum(means) / len(means)
    spread = 0.0
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            spread = max(spread, float(cp.sum((means[i] - means[j]) ** 2).get()))
    return {
        "kind": "gold",
        "target": "rr_extrapolation",
        "gold_id": gold_id,
        "dataset": dataset,
        "prior": prior,
        "dim": dim,
        "kernel": "mala",
        "mala_eta": eta,
        "pilot_scale": scale,
        "pilot_accept": pilot_rate,
        "pilot_rates": {str(k): v for k, v in sorted(rates.items())},
        "chains": chains,
        "burnin": burnin,
        "kept": kept,
        "seeds": seeds,
        "seed_accepts": accepts,
        "gold_mean": [float(v) for v in pooled.get()],
        "seed_means": [[float(v) for v in m.get()] for m in means],
        "gold_spread": spread,
        "elapsed_sec": elapsed,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
    }


def load_gold_cache(path):
    cache = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("kind") == "gold":
                    cache[row["gold_id"]] = row
    return cache


# ----------------------------------------------------------------------------
# Protocol P4: dispersed-start hardened gold (screens_precommit.md,
# "Protocol P4" + Amendment v2 E + Amendment v3 V3-1/V3-3/V3-4/V3-5).
# --gold-only: frozen archived per-cell etas (never recalibrated), registered
# start laws, independent init RandomState (345M block), dynamics streams from
# the 346M map, batch/split-half/post-burn bookkeeping, expanded provenance.
# start-mode zero with no seed bases makes NO new RNG draws and reproduces the
# archived gold rows bit-identically (regression-tested).
# ----------------------------------------------------------------------------

P4_SCHEMA_VERSION = "p4-gold-1"
P4_PRECOMMIT = "campaigns/screens_precommit.md#protocol-p4+amendment-v2E+v3+v4-4"
P4_PRIOR_IDX = {0.1: 0, 0.3: 1}
P4_ARM_IDX = {"zero": 0, "chol": 1, "gauss": 2}
P4_GOLD_OUT_DEFAULT = Path("outputs/sweeps/rr_extrapolation_gold_p4_fd354ef.jsonl")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def p4_cell_idx(dataset, prior):
    # V3-3: wdbc/0.1=0, wdbc/0.3=1, spambase/0.1=2, spambase/0.3=3,
    # adult/0.1=4, adult/0.3=5.
    if dataset not in DATASETS or prior not in P4_PRIOR_IDX:
        raise SystemExit(f"P4 seed map is registered only for {DATASETS} x "
                         f"{sorted(P4_PRIOR_IDX)}; got {dataset}/{prior}")
    return DATASETS.index(dataset) * 2 + P4_PRIOR_IDX[prior]


def p4_init_seed(base, cell_idx, seed_idx, arm_idx, attempt):
    # V4-4 (attempt-specific init streams; identical to V3-3 at attempt=0).
    return base + cell_idx * 100000 + seed_idx * 1000 + attempt * 100 + arm_idx


def p4_dynamics_seed(base, cell_idx, seed_idx, arm_idx, attempt):
    return base + cell_idx * 100000 + seed_idx * 1000 + arm_idx * 100 + attempt * 10


def p4_gold_id(dataset, prior, start_mode, start_scale, chains, burnin, kept, eta,
               dyn_seeds, init_seeds, attempt, data_sha256):
    """Expanded P4 gold identity: start mode/scale/seeds, dynamics seeds, eta,
    chains, burn-in, kept length, attempt, data hash.  The script hash lives in
    the row provenance (script_sha256), not in the identity string."""
    init_part = "none" if init_seeds is None else ",".join(str(s) for s in init_seeds)
    return (f"p4|{dataset}|prior={prior}|arm={start_mode}|scale={start_scale}|chains={chains}"
            f"|burnin={burnin}|kept={kept}|eta={eta!r}|dyn={','.join(str(s) for s in dyn_seeds)}"
            f"|init={init_part}|attempt={attempt}|data={data_sha256[:12]}")


def _import_cgt():
    # Same Newton machinery as the S1 screen (scripts/certified_gamma_table.py).
    try:
        import certified_gamma_table as cgt
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import certified_gamma_table as cgt
    return cgt


def p4_map_chol_from_arrays(x64, y64, prior):
    """theta_MAP (full-Hessian Newton, float64, ||grad|| <= 1e-10) and
    chol(H_MAP^{-1}) for the primary dispersed start law."""
    cgt = _import_cgt()
    theta_map, gnorm, iters = cgt.newton_map(x64, y64, prior)
    mask = np.ones(x64.shape[1])
    mask[0] = 0.0
    _, hess = cgt.grad_hess(theta_map, x64, y64, prior, mask)
    hinv = np.linalg.inv(hess)
    hinv = 0.5 * (hinv + hinv.T)  # symmetrize before Cholesky
    return theta_map, np.linalg.cholesky(hinv), {"newton_grad_norm": float(gnorm),
                                                 "newton_iters": int(iters)}


def p4_map_chol(data_dir, dataset, prior):
    cgt = _import_cgt()
    x64, y64 = cgt.load_dataset(data_dir, dataset)
    return p4_map_chol_from_arrays(x64, y64, prior)


def p4_start_states(start_mode, start_scale, init_seed, chains, dim, map_chol=None):
    """Start states for ONE gold seed run: None = archived zero start (NO RNG
    object constructed, NO draw), else float64 (chains, dim).  Registered draw
    order: rs = np.random.RandomState(init_seed); z = rs.standard_normal(
    (chains, dim)).  chol: X0 = theta_MAP + 2 * chol(H_MAP^{-1}) z (row-wise
    z @ chol^T, cov = 4 H_MAP^{-1}); gauss: X0 = start_scale * z."""
    if start_mode == "zero":
        return None
    rs = np.random.RandomState(init_seed)
    z = rs.standard_normal((chains, dim))
    if start_mode == "chol":
        theta_map, chol_hinv = map_chol
        return theta_map[None, :] + 2.0 * (z @ chol_hinv.T)
    if start_mode == "gauss":
        return start_scale * z
    raise ValueError(f"unknown start mode {start_mode}")


def p4_mcse(pooled_batch_means):
    """V3-1: MCSE_j = sd(all pooled batch means_j) / sqrt(n_batches_total)."""
    bm = np.asarray(pooled_batch_means, dtype=np.float64)
    return bm.std(axis=0, ddof=1) / math.sqrt(bm.shape[0])


# --- V4-2 diagnostics: rank-normalized split-Rhat (bulk + folded, gated on
# max) and bulk ESS (Vehtari et al. 2021, Geyer initial-positive + monotone
# pairs, Stan-style tau floor 1/log10(S)).  _split / _rank_normal / _rhat /
# _bulk_ess / diagnostics are duplicated VERBATIM from
# scripts/s3_poisson_skew_screen.py at commit 3dea85a (sha256 dcee5eb611b7...,
# the copy pinned by Amendment v6 and validated by its 35/35 selftest incl.
# the AR(1) known-ESS contract).  Duplication rather than a shared module
# keeps both scripts conformant with V4-2's "implemented in-script" clause
# and leaves the locked, result-bearing S3 script byte-untouched;
# --selftest-p4 cross-checks the two copies bitwise on random draws.
# scipy (rankdata, ndtri) is bound lazily so archived paths stay scipy-free.

rankdata = None  # bound by _import_diag_deps (scipy.stats.rankdata)
ndtri = None     # bound by _import_diag_deps (scipy.special.ndtri)


def _import_diag_deps():
    global rankdata, ndtri
    if rankdata is None:
        from scipy.special import ndtri as _ndtri
        from scipy.stats import rankdata as _rankdata
        rankdata, ndtri = _rankdata, _ndtri


def _split(draws):
    chains, kept, d = draws.shape
    half = kept // 2
    return np.concatenate([draws[:, :half, :], draws[:, half:2 * half, :]], axis=0)


def _rank_normal(x):
    """x (C,N): pooled average ranks -> z = Phi^{-1}((r - 3/8)/(S + 1/4))."""
    c, n = x.shape
    r = rankdata(x.reshape(-1), method="average")
    return ndtri((r - 0.375) / (c * n + 0.25)).reshape(c, n)


def _rhat(z):
    c, n = z.shape
    cm = z.mean(axis=1)
    w = z.var(axis=1, ddof=1).mean()
    b = n * cm.var(ddof=1)
    var_plus = (n - 1.0) / n * w + b / n
    return float(np.sqrt(var_plus / w))


def _bulk_ess(z):
    c, n = z.shape
    centered = z - z.mean(axis=1, keepdims=True)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(centered, nfft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n].real / n
    chain_var = acov[:, 0] * n / (n - 1.0)
    w = chain_var.mean()
    b_over_n = z.mean(axis=1).var(ddof=1) if c > 1 else 0.0
    var_plus = (n - 1.0) / n * w + b_over_n
    rho = 1.0 - (w - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = rho[: 2 * (n // 2)].reshape(-1, 2).sum(axis=1)
    nonpos = np.nonzero(pairs <= 0)[0]
    k = int(nonpos[0]) if len(nonpos) else len(pairs)
    k = max(k, 1)
    tau = 2.0 * np.minimum.accumulate(pairs[:k]).sum() - 1.0
    s_total = c * n
    tau = max(tau, 1.0 / np.log10(s_total))
    return float(s_total / tau)


def diagnostics(draws):
    """Per coordinate: rank-normalized split-Rhat (bulk, folded, max) and
    bulk ESS on the split chains."""
    split = _split(draws)
    d = draws.shape[2]
    rhat_bulk, rhat_folded, ess = [], [], []
    for j in range(d):
        zj = split[:, :, j]
        z = _rank_normal(zj)
        rhat_bulk.append(_rhat(z))
        ess.append(_bulk_ess(z))
        folded = np.abs(zj - np.median(zj))
        rhat_folded.append(_rhat(_rank_normal(folded)))
    rhat_max = [max(a, b) for a, b in zip(rhat_bulk, rhat_folded)]
    return {"split_rhat_bulk": rhat_bulk, "split_rhat_folded": rhat_folded,
            "split_rhat": rhat_max, "bulk_ess": ess}


def p4_diagnostics(seed_series):
    """V4-2: per-coordinate Rhat = max(rank-normalized split-Rhat, folded)
    and rank-normalized bulk ESS, on the per-step ensemble-mean series
    (n_seeds, kept, dim); the 3 seed-series are split in half inside
    diagnostics() (6 half-chains)."""
    _import_diag_deps()
    return diagnostics(np.asarray(seed_series, dtype=np.float64))


def p4_series_sidecar_path(gold_out, cell_idx, arm_idx, attempt):
    """Registered sidecar naming: <gold_out stem>_series_c{cell}_a{arm}_att{attempt}.npz
    (for the default gold_out this is exactly
    outputs/sweeps/rr_extrapolation_gold_p4_fd354ef_series_c*_a*_att*.npz)."""
    return gold_out.parent / f"{gold_out.stem}_series_c{cell_idx}_a{arm_idx}_att{attempt}.npz"


def p4_gold_seed_run(seed, eta, burnin, kept, chains, dim, prior, data_x, data_y,
                     chunk_rows, x0, batch_len):
    """One P4 gold seed run.  With x0=None this is draw-for-draw identical to
    gold_seed_run: cp.random.seed(seed), then per step exactly one
    standard_normal (1, chains, dim) float32 plus one uniform (1, chains)
    float32 inside mala_step -- same call sites, order, and count; the batch /
    split-half / post-burn / series bookkeeping consumes no randomness
    (accumulation and .get() reads only).  Returns (mean, accept_overall,
    accept_postburn, batch_means, series): batch_means host-side
    (kept // batch_len, dim); series host-side (kept, dim) float64 -- the
    V4-2 per-step ensemble-mean series whose batch means define the MCSE."""
    assert kept % batch_len == 0, (kept, batch_len)
    cp.random.seed(seed)
    if x0 is None:
        x = cp.zeros((1, chains, dim), dtype=cp.float32)
    else:
        x = cp.asarray(np.asarray(x0, dtype=np.float32).reshape(1, chains, dim))
    u, g = potential_and_grad(x, data_x, data_y, prior, chunk_rows)
    acc = cp.zeros(dim, dtype=cp.float64)
    accept_acc = cp.zeros((), dtype=cp.float32)
    batch_acc = cp.zeros((kept // batch_len, dim), dtype=cp.float64)
    series = cp.zeros((kept, dim), dtype=cp.float64)
    accept_at_burn = 0.0
    for step in range(burnin + kept):
        noise = cp.random.standard_normal(x.shape, dtype=cp.float32)
        x, u, g = mala_step(x, u, g, noise, eta, prior, data_x, data_y, chunk_rows, accept_acc)
        if step == burnin - 1:
            accept_at_burn = float(accept_acc.get())
        if step >= burnin:
            xm = x.mean(axis=(0, 1), dtype=cp.float64)
            acc += xm
            batch_acc[(step - burnin) // batch_len] += xm
            series[step - burnin] = xm
    # Overall acceptance uses the archived expression exactly (float32 divide
    # then float()) so the value is bit-identical to gold_seed_run's.
    accept_overall = float((accept_acc / (burnin + kept)).get())
    accept_postburn = (float(accept_acc.get()) - accept_at_burn) / kept
    return (acc / kept, accept_overall, accept_postburn,
            (batch_acc / batch_len).get(), series.get())


def run_gold_only(args):
    """P4 --gold-only: gold references only (no replicate-file checks, no
    13-estimator campaign).  Frozen archived etas; never writes the archived
    gold file.  Accepted-run output path is P4_GOLD_OUT_DEFAULT; tests must
    pass a scratch --gold-out."""
    start_mode = args.start_mode
    arm_idx = P4_ARM_IDX[start_mode]
    attempt = args.attempt
    if start_mode == "zero":
        if args.start_scale != 0.0:
            sys.exit("start-mode zero requires --start-scale 0.0 (no draw is made)")
        eff_scale = 0.0
    elif start_mode == "chol":
        # The registered law fixes the factor: X0 = theta_MAP + 2 chol(H^-1) z.
        if args.start_scale not in (0.0, 2.0):
            sys.exit("start-mode chol uses the registered fixed factor 2.0; "
                     "leave --start-scale unset or pass 2.0")
        if args.start_seed_base is None:
            sys.exit("start-mode chol requires --start-seed-base (V3-3 345M init streams)")
        eff_scale = 2.0
    else:  # gauss
        if args.start_seed_base is None:
            sys.exit("start-mode gauss requires --start-seed-base (V3-3 345M init streams)")
        if args.start_scale <= 0.0:
            sys.exit("start-mode gauss requires --start-scale > 0")
        eff_scale = args.start_scale
    if attempt == 1 and args.dynamics_seed_base is None:
        sys.exit("--attempt 1 (remediation) requires --dynamics-seed-base (V3-3 346M map)")

    archived_gold = args.out_dir / f"{args.prefix}_gold.jsonl"
    gold_out = args.gold_out
    if gold_out.resolve() == archived_gold.resolve():
        sys.exit(f"refusing to write P4 gold rows into the archived gold file {archived_gold}")
    frozen_path = args.frozen_gold if args.frozen_gold is not None else archived_gold
    if not frozen_path.exists():
        sys.exit(f"frozen gold file {frozen_path} not found (archived etas are required; "
                 "recalibration is forbidden)")
    if gold_out.resolve() == frozen_path.resolve():
        sys.exit(f"refusing to append P4 gold rows to the frozen-eta source {frozen_path}")

    frozen = {}
    with frozen_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "gold":
                continue
            key = (row["dataset"], row["prior"])
            if key in frozen:
                sys.exit(f"duplicate gold row for cell {key} in {frozen_path}")
            frozen[key] = row

    gold_out.parent.mkdir(parents=True, exist_ok=True)
    cache = load_gold_cache(gold_out)
    script_sha = sha256_file(Path(__file__))
    print(f"P4 gold-only: arm={start_mode} scale={eff_scale} attempt={attempt} "
          f"start_base={args.start_seed_base} dyn_base={args.dynamics_seed_base} "
          f"batch_len={args.batch_len} frozen etas from {frozen_path} -> {gold_out} "
          f"(accepted runs remain gated per Amendment V3-7)", flush=True)

    for ds in args.datasets:
        data_x, data_y = load_dataset(args.data_dir, ds)
        dim = int(data_x.shape[1])
        data_sha = sha256_file(args.data_dir / f"{ds}.npz")
        for prior in args.priors:
            key = (ds, prior)
            if key not in frozen:
                sys.exit(f"no frozen gold row for cell {key} in {frozen_path}")
            fr = frozen[key]
            if fr["dim"] != dim:
                sys.exit(f"dim mismatch for {key}: data {dim} vs frozen row {fr['dim']}")
            eta = fr["mala_eta"]
            chains, burnin, kept = fr["chains"], fr["burnin"], fr["kept"]
            if attempt == 1:
                burnin, kept = 2 * burnin, 2 * kept  # V3-4 locked remediation arithmetic
            if kept % args.batch_len != 0 or (kept // args.batch_len) % 2 != 0:
                sys.exit(f"kept={kept} needs an even number of full --batch-len={args.batch_len} "
                         "batches (nonoverlapping batch means + split halves)")
            cell = p4_cell_idx(ds, prior)
            if args.dynamics_seed_base is None:
                dyn_seeds = [GOLD_SEED_BASE + DS_OFF[ds] + int(round(prior * 1000)) + j
                             for j in range(GOLD_N_SEEDS)]
                if dyn_seeds != fr["seeds"]:
                    sys.exit(f"archived-mode dynamics seeds {dyn_seeds} != frozen row "
                             f"seeds {fr['seeds']} for {key}")
            else:
                dyn_seeds = [p4_dynamics_seed(args.dynamics_seed_base, cell, j, arm_idx, attempt)
                             for j in range(GOLD_N_SEEDS)]
            # Arm 0 (zero start) is deterministic: its init stream is
            # reserved but never consumed; no RandomState is constructed.
            init_seeds = None if start_mode == "zero" else \
                [p4_init_seed(args.start_seed_base, cell, j, arm_idx, attempt)
                 for j in range(GOLD_N_SEEDS)]
            gold_id = p4_gold_id(ds, prior, start_mode, eff_scale, chains, burnin, kept,
                                 eta, dyn_seeds, init_seeds, attempt, data_sha)
            if gold_id in cache:
                print(f"p4 gold cache hit: {gold_id}", flush=True)
                continue
            sidecar = p4_series_sidecar_path(gold_out, cell, arm_idx, attempt)
            if sidecar.exists():
                sys.exit(f"refusing to overwrite existing series sidecar {sidecar} "
                         "(no matching cached gold row; delete it to recompute)")
            map_chol = None
            newton_meta = {}
            if start_mode == "chol":
                theta_map, chol_hinv, newton_meta = p4_map_chol(args.data_dir, ds, prior)
                map_chol = (theta_map, chol_hinv)

            t0 = time.perf_counter()
            means, accepts, accepts_post, seed_bms, seed_series = [], [], [], [], []
            for j in range(GOLD_N_SEEDS):
                x0 = p4_start_states(start_mode, eff_scale,
                                     None if init_seeds is None else init_seeds[j],
                                     chains, dim, map_chol)
                m, a, ap, bm, ser = p4_gold_seed_run(dyn_seeds[j], eta, burnin, kept, chains,
                                                     dim, prior, data_x, data_y,
                                                     args.chunk_rows, x0, args.batch_len)
                means.append(m)
                accepts.append(a)
                accepts_post.append(ap)
                seed_bms.append(np.asarray(bm, dtype=np.float64))
                seed_series.append(np.asarray(ser, dtype=np.float64))
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - t0
            pooled = sum(means) / len(means)
            spread = 0.0
            for i in range(len(means)):
                for j2 in range(i + 1, len(means)):
                    spread = max(spread, float(cp.sum((means[i] - means[j2]) ** 2).get()))
            nb = kept // args.batch_len
            half = nb // 2
            # Halves are exact means over the first/second half of kept draws:
            # equal-length batches partition the kept window evenly.
            mcse = p4_mcse(np.concatenate(seed_bms, axis=0))
            # V4-2 diagnostics on the per-step ensemble-mean series; the raw
            # series goes to the registered .npz sidecar (one per
            # cell/arm/attempt), sha256-recorded in the row.
            series_all = np.stack(seed_series)
            diag = p4_diagnostics(series_all)
            rhat = [float(v) for v in diag["split_rhat"]]
            ess = [float(v) for v in diag["bulk_ess"]]
            np.savez_compressed(
                sidecar,
                series=series_all,
                dynamics_seeds=np.asarray(dyn_seeds, dtype=np.int64),
                gold_id=np.asarray(gold_id),
                dataset=np.asarray(ds),
                prior=np.asarray(prior),
                cell_idx=np.asarray(cell),
                arm_idx=np.asarray(arm_idx),
                attempt=np.asarray(attempt),
                burnin=np.asarray(burnin),
                kept=np.asarray(kept),
                batch_len=np.asarray(args.batch_len),
            )
            sidecar_sha = sha256_file(sidecar)
            row = {
                "kind": "gold",
                "target": "rr_extrapolation",
                "gold_id": gold_id,
                "schema_version": P4_SCHEMA_VERSION,
                "precommit": P4_PRECOMMIT,
                "dataset": ds,
                "prior": prior,
                "dim": dim,
                "kernel": "mala",
                "mala_eta": eta,
                "eta_source": f"frozen:{frozen_path}",
                "frozen_gold_id": fr["gold_id"],
                "pilot_scale": fr["pilot_scale"],
                "pilot_accept": fr["pilot_accept"],
                "pilot_rates": fr["pilot_rates"],
                "chains": chains,
                "burnin": burnin,
                "kept": kept,
                "seeds": dyn_seeds,
                "dynamics_seeds": dyn_seeds,
                "init_seeds": init_seeds,
                "start_mode": start_mode,
                "start_scale": eff_scale,
                "start_seed_base": args.start_seed_base,
                "dynamics_seed_base": args.dynamics_seed_base,
                "attempt": attempt,
                "cell_idx": cell,
                "arm_idx": arm_idx,
                "seed_accepts": accepts,
                "seed_accepts_postburn": accepts_post,
                "gold_mean": [float(v) for v in pooled.get()],
                "seed_means": [[float(v) for v in m.get()] for m in means],
                "gold_spread": spread,
                "batch_len": args.batch_len,
                "n_batches_per_seed": nb,
                "n_batches_total": nb * GOLD_N_SEEDS,
                "seed_batch_means": [[[float(v) for v in b] for b in bm] for bm in seed_bms],
                "seed_split_half_first": [[float(v) for v in bm[:half].mean(axis=0)]
                                          for bm in seed_bms],
                "seed_split_half_second": [[float(v) for v in bm[half:].mean(axis=0)]
                                           for bm in seed_bms],
                "mcse": [float(v) for v in mcse],
                "split_rhat": rhat,
                "split_rhat_bulk": [float(v) for v in diag["split_rhat_bulk"]],
                "split_rhat_folded": [float(v) for v in diag["split_rhat_folded"]],
                "bulk_ess": ess,
                "rhat_max": max(rhat),
                "ess_min": min(ess),
                "series_sidecar": str(sidecar),
                "series_sidecar_sha256": sidecar_sha,
                **newton_meta,
                "data_sha256": data_sha,
                "script_sha256": script_sha,
                "argv": sys.argv,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_sec": elapsed,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
                "git_hash": git_hash(),
            }
            with gold_out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
            cache[gold_id] = row
            print(f"p4 gold [{ds} prior={prior} arm={start_mode} attempt={attempt}]: eta={eta:.4g} "
                  f"accepts={[round(a, 3) for a in accepts]} "
                  f"postburn={[round(a, 3) for a in accepts_post]} spread={spread:.3e} "
                  f"max_mcse={float(np.max(mcse)):.3e} rhat_max={max(rhat):.4f} "
                  f"ess_min={min(ess):.0f} sidecar={sidecar.name} ({elapsed:.1f}s)", flush=True)


# --- CPU numpy stand-in for the cupy API surface used above (P4 unit tests
# only; --selftest-p4 never imports cupy, so it can never touch a GPU). ------

class _CpuArray(np.ndarray):
    """ndarray with cupy-style .get(); reductions re-wrapped so full
    reductions keep .get() (numpy returns plain scalars there)."""

    def get(self):
        return np.asarray(self).copy()

    def sum(self, *a, **k):
        return np.asarray(super().sum(*a, **k)).view(_CpuArray)

    def mean(self, *a, **k):
        return np.asarray(super().mean(*a, **k)).view(_CpuArray)


class _CpuRandom:
    seed = staticmethod(np.random.seed)

    @staticmethod
    def standard_normal(size, dtype=np.float64):
        return np.random.standard_normal(size).astype(dtype, copy=False).view(_CpuArray)

    @staticmethod
    def random(size=None, dtype=np.float64):
        return np.random.random(size).astype(dtype, copy=False).view(_CpuArray)


class _CpuCupyShim:
    """Minimal numpy-backed cupy substitute for the CPU-only P4 selftest; the
    GPU path is untouched.  Only the API used by the gold path exists."""
    float32 = np.float32
    float64 = np.float64
    random = _CpuRandom

    class cuda:
        class Stream:
            class null:
                synchronize = staticmethod(lambda: None)

    def __getattr__(self, name):
        f = getattr(np, name)
        if not callable(f):
            return f

        def wrapped(*a, **k):
            out = f(*a, **k)
            if isinstance(out, (np.ndarray, np.generic)):
                return np.asarray(out).view(_CpuArray)
            return out
        return wrapped


def _np_state_equal(a, b):
    return (a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
            and a[3] == b[3] and a[4] == b[4])


def p4_seed_map_lines():
    """Full seed map over all cells/seeds/arms/attempts (published bases
    345,000,000 init / 346,000,000 dynamics; V3-3 dynamics formula, V4-4
    attempt-specific init formula, identical to V3-3 init at attempt=0)."""
    lines = ["P4 seed map: init = 345000000 + cell*100000 + seed*1000 + attempt*100 + arm (V4-4); "
             "dyn = 346000000 + cell*100000 + seed*1000 + arm*100 + attempt*10 (V3-3); "
             "arms: 0=zero(control; init stream reserved, never consumed) "
             "1=chol(primary dispersed) 2=gauss(N(0,I) comparability)"]
    init_all, dyn_all = [], []
    for ds in DATASETS:
        for pr in sorted(P4_PRIOR_IDX):
            c = p4_cell_idx(ds, pr)
            for s in range(GOLD_N_SEEDS):
                init0 = [p4_init_seed(345000000, c, s, a, 0) for a in range(3)]
                init1 = [p4_init_seed(345000000, c, s, a, 1) for a in range(3)]
                dyn0 = [p4_dynamics_seed(346000000, c, s, a, 0) for a in range(3)]
                dyn1 = [p4_dynamics_seed(346000000, c, s, a, 1) for a in range(3)]
                init_all += init0 + init1
                dyn_all += dyn0 + dyn1
                lines.append(f"cell{c} {ds}/{pr} seed{s}: "
                             f"init@att0(z,c,g)={init0[0]}(unused),{init0[1]},{init0[2]} "
                             f"init@att1(z,c,g)={init1[0]}(unused),{init1[1]},{init1[2]} "
                             f"dyn@att0={dyn0[0]},{dyn0[1]},{dyn0[2]} "
                             f"dyn@att1={dyn1[0]},{dyn1[1]},{dyn1[2]}")
    return lines, init_all, dyn_all


def selftest_p4():
    """CPU-only P4 unit checks (numpy shim; cupy is never imported): V3-3 seed
    formulas + in-block disjointness, scale-0 no-draw + archived-path
    equivalence, chol/gauss start laws, batch / split-half / MCSE bookkeeping,
    expanded gold_id fields."""
    global cp
    cp = _CpuCupyShim()
    checks = {}

    # (A) Seed map: registered cell_idx mapping, full printout, disjointness.
    expected_cells = {("wdbc", 0.1): 0, ("wdbc", 0.3): 1, ("spambase", 0.1): 2,
                      ("spambase", 0.3): 3, ("adult", 0.1): 4, ("adult", 0.3): 5}
    checks["seed_map_cell_idx_registered"] = all(
        p4_cell_idx(ds, pr) == v for (ds, pr), v in expected_cells.items())
    lines, init_all, dyn_all = p4_seed_map_lines()
    print("\n".join(lines))
    map_digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
    print(f"seed-map sha256: {map_digest}")
    checks["seed_map_init_108_distinct_in_345M_block"] = (
        len(init_all) == 108 and len(set(init_all)) == 108
        and min(init_all) == 345000000 and max(init_all) == 345502102
        and all(345000000 <= s < 346000000 for s in init_all))
    checks["seed_map_dyn_108_distinct_in_346M_block"] = (
        len(dyn_all) == 108 and len(set(dyn_all)) == 108
        and min(dyn_all) == 346000000 and max(dyn_all) == 346502210
        and all(346000000 <= s < 347000000 for s in dyn_all))
    checks["seed_map_blocks_disjoint"] = not set(init_all) & set(dyn_all)

    # Tiny synthetic logistic cell (CPU shim; seconds).
    rs = np.random.RandomState(20260717)
    n_pts, d_feat = 48, 3
    xf = rs.standard_normal((n_pts, d_feat)).astype(np.float32)
    logits = 0.8 * xf[:, 0] - 0.5 * xf[:, 1] + 0.3
    y_np = (rs.random_sample(n_pts) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    data_x = cp.asarray(add_intercept(xf), dtype=cp.float32)
    data_y = cp.asarray(y_np, dtype=cp.float32)
    dim, prior, eta = d_feat + 1, 0.3, 0.4
    chains, burnin, kept, batch_len = 6, 64, 240, 8

    # (B) scale-0 regression: zero-mode builder touches no RNG; the P4 runner
    # with x0=None matches the archived gold_seed_run bit-for-bit AND leaves
    # the RNG in the identical end state (same draw count incl. gauss cache).
    st0 = np.random.get_state()
    x0 = p4_start_states("zero", 0.0, None, chains, dim)
    checks["scale0_start_builder_makes_no_rng_draw"] = (
        x0 is None and _np_state_equal(st0, np.random.get_state()))
    m_ref, a_ref = gold_seed_run(4242, eta, burnin, kept, chains, dim, prior,
                                 data_x, data_y, 1024)
    st_ref = np.random.get_state()
    m_new, a_new, a_post, bm, ser1 = p4_gold_seed_run(4242, eta, burnin, kept, chains, dim,
                                                      prior, data_x, data_y, 1024, None,
                                                      batch_len)
    st_new = np.random.get_state()
    checks["scale0_bitwise_equals_archived_run_and_rng_state"] = (
        np.array_equal(m_ref.get(), m_new.get()) and a_ref == a_new
        and _np_state_equal(st_ref, st_new))
    checks["postburn_acceptance_finite_in_(0,1]"] = bool(
        math.isfinite(a_post) and 0.0 < a_post <= 1.0)
    print(f"scale0 check: |mean|max={float(np.max(np.abs(m_ref.get()))):.4f} "
          f"accept={a_ref:.3f} postburn={a_post:.3f}")

    # (C) chol start law: mean theta_MAP, cov 4 H^-1, deterministic per seed,
    # different seeds differ; gauss law: mean 0, var scale^2.
    x64 = add_intercept(xf).astype(np.float64)
    y64 = y_np.astype(np.float64)
    theta_map, chol_hinv, meta = p4_map_chol_from_arrays(x64, y64, prior)
    hinv4 = 4.0 * (chol_hinv @ chol_hinv.T)
    big = 4000
    xs = p4_start_states("chol", 2.0, 345000001, big, dim, (theta_map, chol_hinv))
    xs2 = p4_start_states("chol", 2.0, 345000001, big, dim, (theta_map, chol_hinv))
    xs3 = p4_start_states("chol", 2.0, 345001001, big, dim, (theta_map, chol_hinv))
    se = np.sqrt(np.diag(hinv4) / big)
    mean_off = np.abs(xs.mean(axis=0) - theta_map)
    cov_rel = np.linalg.norm(np.cov(xs.T) - hinv4) / np.linalg.norm(hinv4)
    checks["chol_start_law_shape_mean_cov_deterministic"] = bool(
        xs.shape == (big, dim) and np.array_equal(xs, xs2)
        and not np.array_equal(xs, xs3) and np.all(mean_off <= 5.0 * se)
        and cov_rel < 0.15 and meta["newton_grad_norm"] <= 1e-10)
    gz = p4_start_states("gauss", 1.5, 345000002, 5000, dim)
    checks["gauss_start_law_shape_mean_var"] = bool(
        gz.shape == (5000, dim)
        and np.all(np.abs(gz.mean(axis=0)) <= 5.0 * 1.5 / math.sqrt(5000))
        and np.all(np.abs(gz.var(axis=0, ddof=1) / 2.25 - 1.0) < 0.1))
    print(f"chol law: max|mean-theta_MAP|/SE={float(np.max(mean_off / se)):.2f} "
          f"cov relerr={cov_rel:.3f} newton gnorm={meta['newton_grad_norm']:.2e} "
          f"iters={meta['newton_iters']}")

    # (D) Batch-mean and split-half bookkeeping.
    nb, half = kept // batch_len, kept // batch_len // 2
    checks["batch_means_shape_and_consistency_with_run_mean"] = bool(
        bm.shape == (nb, dim)
        and np.allclose(bm.mean(axis=0), m_new.get(), rtol=1e-10, atol=1e-12))
    first, second = bm[:half].mean(axis=0), bm[half:].mean(axis=0)
    checks["split_half_consistency_with_run_mean"] = bool(
        np.allclose(0.5 * (first + second), m_new.get(), rtol=1e-10, atol=1e-12))
    # Known-variance MCSE: iid N(0.9, 1.7^2), 3 pseudo-seeds x 20 batches of
    # 100 -> MCSE tracks sigma/sqrt(total kept) (60 batches: sd ~9% per coord).
    sig, kept_s, bl = 1.7, 2000, 100
    rs2 = np.random.RandomState(3)
    pooled_bm = [
        (0.9 + sig * rs2.standard_normal((kept_s, 32))).reshape(kept_s // bl, bl, 32).mean(axis=1)
        for _ in range(3)]
    mcse = p4_mcse(np.concatenate(pooled_bm, axis=0))
    ratio = mcse / (sig / math.sqrt(3 * kept_s))
    checks["mcse_within_tolerance_on_known_variance_draws"] = bool(
        0.9 <= float(ratio.mean()) <= 1.1 and np.all((ratio >= 0.55) & (ratio <= 1.55)))
    print(f"mcse check: mean ratio={float(ratio.mean()):.3f} "
          f"range=[{float(ratio.min()):.3f},{float(ratio.max()):.3f}] (expect ~1)")

    # (E) Expanded gold_id fields.
    gid = p4_gold_id("wdbc", 0.1, "chol", 2.0, 64, 5000, 30000, 3.2,
                     [346000100, 346001100, 346002100],
                     [345000001, 345001001, 345002001], 0, "a" * 64)
    need = ["p4|wdbc|prior=0.1", "arm=chol", "scale=2.0", "chains=64", "burnin=5000",
            "kept=30000", "eta=3.2", "dyn=346000100,346001100,346002100",
            "init=345000001,345001001,345002001", "attempt=0", "data=aaaaaaaaaaaa"]
    gid0 = p4_gold_id("adult", 0.3, "zero", 0.0, 64, 5000, 30000, 0.6545454545454544,
                      [12100300, 12100301, 12100302], None, 0, "b" * 64)
    checks["gold_id_encodes_registered_fields"] = bool(
        all(t in gid for t in need) and "init=none" in gid0
        and "eta=0.6545454545454544" in gid0)
    print(f"gold_id sample: {gid}")

    # (F) V4-2 diagnostics: iid sanity, shifted-chain detection, AR(1)
    # known-ESS contract, 6-half-chain end-to-end run, and a bitwise
    # cross-check against the authoritative S3 source copy.
    rs3 = np.random.RandomState(555)
    iid = rs3.standard_normal((3, 2000, 3))
    di = p4_diagnostics(iid)
    s_total = 3 * 2000
    checks["diag_iid_rhat_near_1_and_ess_near_S"] = bool(
        max(di["split_rhat"]) <= 1.01
        and 0.7 * s_total <= min(di["bulk_ess"])
        and max(di["bulk_ess"]) <= 1.35 * s_total)
    shifted = iid + np.linspace(-1.0, 1.0, 3)[:, None, None]
    dshift = p4_diagnostics(shifted)
    checks["diag_shifted_chains_detected"] = bool(max(dshift["split_rhat"]) > 1.05)
    phi, n_ar = 0.95, 4000
    innov = rs3.standard_normal((3, n_ar, 1)) * math.sqrt(1 - phi ** 2)
    ar1 = np.empty_like(innov)
    state = rs3.standard_normal((3, 1))
    for t in range(n_ar):
        state = phi * state + innov[:, t, :]
        ar1[:, t, :] = state
    dar = p4_diagnostics(ar1)
    ess_true = 3 * n_ar * (1 - phi) / (1 + phi)
    ratio_ess = dar["bulk_ess"][0] / ess_true
    checks["diag_ar1_ess_within_0.3_3x_known_value"] = bool(0.3 <= ratio_ess <= 3.0)
    print(f"diag: iid rhat_max={max(di['split_rhat']):.4f} "
          f"ess/S in [{min(di['bulk_ess']) / s_total:.2f},{max(di['bulk_ess']) / s_total:.2f}] | "
          f"shifted rhat_max={max(dshift['split_rhat']):.2f} | "
          f"AR(1) ess={dar['bulk_ess'][0]:.0f} vs true {ess_true:.0f} (ratio {ratio_ess:.2f})")

    # End-to-end: 3 dynamics seeds -> (3, kept, dim) series -> 6 half-chains;
    # the recorded series must reproduce the batch means bookkeeping.
    m2, a2, ap2, bm2, ser2 = p4_gold_seed_run(4243, eta, burnin, kept, chains, dim, prior,
                                              data_x, data_y, 1024, None, batch_len)
    m3, a3, ap3, bm3, ser3 = p4_gold_seed_run(4244, eta, burnin, kept, chains, dim, prior,
                                              data_x, data_y, 1024, None, batch_len)
    series3 = np.stack([ser1, ser2, ser3])
    d3 = p4_diagnostics(series3)
    rhat_max3 = max(d3["split_rhat"])
    ess_min3 = min(d3["bulk_ess"])
    nb = kept // batch_len
    series_bm_ok = all(
        np.allclose(s.reshape(nb, batch_len, dim).mean(axis=1), b, rtol=1e-10, atol=1e-12)
        for s, b in [(ser1, bm), (ser2, bm2), (ser3, bm3)])
    checks["diag_endtoend_6halfchains_sane_series_matches_batches"] = bool(
        series3.shape == (3, kept, dim)
        and all(math.isfinite(v) for v in d3["split_rhat"] + d3["bulk_ess"])
        and 0.95 < rhat_max3 < 2.0 and 0.0 < ess_min3 <= 1.35 * 3 * kept
        and series_bm_ok)
    print(f"diag end-to-end: rhat_max={rhat_max3:.3f} ess_min={ess_min3:.0f} "
          f"(3 seeds x kept={kept})")

    # Bitwise cross-check vs the authoritative source copy (anti-drift).
    # The source copy (s3_poisson_skew_screen.py) is not part of this
    # repository; the cross-check runs only when that file is present.
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import s3_poisson_skew_screen as s3mod
    except ImportError:
        s3mod = None
        print("source copy s3_poisson_skew_screen.py not present: "
              "bitwise cross-check skipped")
    if s3mod is not None:
        xchk = rs3.standard_normal((3, 800, 4))
        mine, theirs = p4_diagnostics(xchk), s3mod.diagnostics(xchk)
        checks["diag_bitwise_matches_s3_source_copy"] = bool(
            all(mine[k] == theirs[k]
                for k in ["split_rhat_bulk", "split_rhat_folded", "split_rhat", "bulk_ess"]))

    print("\nselftest-p4 checks:")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    verdict = all(checks.values())
    print(f"selftest-p4 {'PASS' if verdict else 'FAIL'} "
          f"({sum(bool(v) for v in checks.values())}/{len(checks)})")
    return 0 if verdict else 1


# ----------------------------------------------------------------------------
# Estimators.
# ----------------------------------------------------------------------------

def estimator_specs(eta_c, budget):
    """13 rows per cell: for each pair (primary, secondary): 4 single-level
    (iid/zs x eta1/eta2, full budget) + 2 RR (iid/zs); plus 1 MALA context."""
    specs = []
    for pair, e1, e2 in [("primary", 4 * eta_c, 2 * eta_c), ("secondary", eta_c, eta_c / 2)]:
        for coupling in ["iid", "zero_sum"]:
            for level, eta in [("eta1", e1), ("eta2", e2)]:
                specs.append({"estimator": "single", "pair": pair, "coupling": coupling,
                              "level": level, "eta1": eta, "eta2": None,
                              "steps1": budget, "steps2": None, "kernel": "ula"})
        for coupling in ["iid", "zero_sum"]:
            specs.append({"estimator": "rr", "pair": pair, "coupling": coupling,
                          "level": None, "eta1": e1, "eta2": e2,
                          "steps1": budget // 3, "steps2": 2 * budget // 3, "kernel": "ula"})
    specs.append({"estimator": "mala", "pair": "context", "coupling": "zero_sum",
                  "level": None, "eta1": None, "eta2": None,
                  "steps1": budget, "steps2": None, "kernel": "mala"})
    return specs


def run_estimator(spec, seed, gold_row, n_ens, chains, dim, prior, data_x, data_y, chunk_rows):
    gold = cp.asarray(gold_row["gold_mean"], dtype=cp.float64)
    eta1 = spec["eta1"] if spec["eta1"] is not None else gold_row["mala_eta"]
    accept_rate = None
    warm_up(spec["kernel"], spec["coupling"], eta1, chains, dim, prior, data_x, data_y, chunk_rows)
    t0 = time.perf_counter()
    if spec["estimator"] == "single":
        theta = ula_time_avg(seed, spec["coupling"], eta1, spec["steps1"], n_ens, chains, dim, prior, data_x, data_y, chunk_rows)
    elif spec["estimator"] == "rr":
        # Independent noise between levels (distinct seeds); each level from
        # zero starts with its own 50% burn-in.  theta_RR = 2*avg(eta2) - avg(eta1).
        avg1 = ula_time_avg(seed, spec["coupling"], spec["eta1"], spec["steps1"], n_ens, chains, dim, prior, data_x, data_y, chunk_rows)
        avg2 = ula_time_avg(seed + 250, spec["coupling"], spec["eta2"], spec["steps2"], n_ens, chains, dim, prior, data_x, data_y, chunk_rows)
        theta = 2.0 * avg2 - avg1
    else:  # mala context at the gold pilot-calibrated step
        theta, accept_rate = mala_time_avg(seed, spec["coupling"], eta1, spec["steps1"], n_ens, chains, dim, prior, data_x, data_y, chunk_rows)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    err = theta - gold[None, :]
    sq = cp.sum(err * err, axis=1)
    finite = cp.isfinite(sq)
    n_finite = int(finite.sum().get())
    if n_finite >= 2:
        sq_f = sq[finite]
        theta_f = theta[finite]
        theta_bar = theta_f.mean(axis=0)
        mse_median = float(cp.median(sq_f).get())
        mse_mean = float(sq_f.mean().get())
        bias_sq = float(cp.sum((theta_bar - gold) ** 2).get())
        trace_var = float(cp.mean(cp.sum((theta_f - theta_bar[None, :]) ** 2, axis=1)).get())
    else:
        mse_median = mse_mean = bias_sq = trace_var = None
    return {
        "kind": "estimator",
        "target": "rr_extrapolation",
        "estimator": spec["estimator"],
        "pair": spec["pair"],
        "coupling": spec["coupling"],
        "level": spec["level"],
        "kernel": spec["kernel"],
        "eta1": eta1,
        "eta2": spec["eta2"],
        "steps1": spec["steps1"],
        "steps2": spec["steps2"],
        "accept_rate": accept_rate,
        "seed": seed,
        "n_ens": n_ens,
        "chains": chains,
        "n_finite": n_finite,
        "mse_gold_median": mse_median,
        "mse_gold_mean": mse_mean,
        "bias_sq_gold": bias_sq,
        "trace_var": trace_var,
        "elapsed_sec": elapsed,
        "elapsed_sec_per_ensemble": elapsed / n_ens,
        "gold_id": gold_row["gold_id"],
        "gold_spread": gold_row["gold_spread"],
        "gold_mala_eta": gold_row["mala_eta"],
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
    }


# ----------------------------------------------------------------------------
# Summaries and pre-commit bar evaluation.
# ----------------------------------------------------------------------------

def fmt(v):
    return "diverged" if v is None else f"{v:.3e}"


def print_replicate_summary(rows, replicate):
    print(f"\n=== replicate {replicate} per-cell summary (primary pair, median MSE to gold) ===")
    cells = sorted({(r["dataset"], r["prior"]) for r in rows if r["kind"] == "estimator"})
    for ds, pr in cells:
        sel = {}
        for r in rows:
            if r["kind"] != "estimator" or r["dataset"] != ds or r["prior"] != pr:
                continue
            key = (r["pair"], r["estimator"], r["coupling"], r.get("level"))
            sel[key] = r
        for pair in ["primary", "secondary"]:
            singles = {(c, l): sel.get((pair, "single", c, l)) for c in ["iid", "zero_sum"] for l in ["eta1", "eta2"]}
            zs_rr = sel.get((pair, "rr", "zero_sum", None))
            iid_rr = sel.get((pair, "rr", "iid", None))
            finite_singles = {k: v for k, v in singles.items() if v and v["mse_gold_median"] is not None}
            best_key = min(finite_singles, key=lambda k: finite_singles[k]["mse_gold_median"]) if finite_singles else None
            best = finite_singles[best_key] if best_key else None
            zs_rr_mse = zs_rr["mse_gold_median"] if zs_rr else None
            line = (f"[{ds} prior={pr}] {pair:9s} zsRR={fmt(zs_rr_mse)} "
                    f"iidRR={fmt(iid_rr['mse_gold_median'] if iid_rr else None)} "
                    f"bestSingle={fmt(best['mse_gold_median'] if best else None)}"
                    f"{'(' + best_key[0] + '@' + best_key[1] + ')' if best_key else ''}")
            if best and zs_rr_mse:
                line += f" best/zsRR={best['mse_gold_median'] / zs_rr_mse:.2f}x"
            if iid_rr and iid_rr["mse_gold_median"] and zs_rr_mse:
                line += f" iidRR/zsRR={iid_rr['mse_gold_median'] / zs_rr_mse:.2f}x"
            print(line)
            if pair == "primary" and zs_rr and best:
                print(f"    split zsRR: bias2={fmt(zs_rr['bias_sq_gold'])} var={fmt(zs_rr['trace_var'])} | "
                      f"bestSingle: bias2={fmt(best['bias_sq_gold'])} var={fmt(best['trace_var'])}")
        ctx = sel.get(("context", "mala", "zero_sum", None))
        if ctx:
            print(f"    context zs-MALA: mse={fmt(ctx['mse_gold_median'])} accept={ctx['accept_rate']:.3f} eta={ctx['eta1']:.3f}")


def evaluate(args):
    rows = []
    for rep in args.replicates:
        path = args.out_dir / f"{args.prefix}_rep{rep}.jsonl"
        if not path.exists():
            print(f"missing {path}")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["replicate_file"] = rep
                rows.append(r)
    est = [r for r in rows if r.get("kind") == "estimator"]
    golds = {r["gold_id"]: r for r in rows if r.get("kind") == "gold"}
    if not est:
        print("no estimator rows found")
        return

    def med(vals):
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    def agg(ds, pr, pair, estimator, coupling, level=None, field="mse_gold_median"):
        vals = [r[field] for r in est
                if r["dataset"] == ds and r["prior"] == pr and r["pair"] == pair
                and r["estimator"] == estimator and r["coupling"] == coupling and r.get("level") == level]
        return med(vals)

    cells = sorted({(r["dataset"], r["prior"]) for r in est})
    print("\n===== pre-commit bar evaluation (median over replicates of per-replicate median MSE to gold) =====")
    for gid, g in sorted(golds.items()):
        print(f"gold {gid}: eta={g['mala_eta']:.3f} pilot_accept={g['pilot_accept']:.3f} "
              f"seed_accepts={[round(a, 3) for a in g['seed_accepts']]} spread={g['gold_spread']:.3e}")

    verdict = {}
    for pair in ["primary", "secondary"]:
        print(f"\n--- pair: {pair} ---")
        per_cell = {}
        for ds, pr in cells:
            singles = {}
            for c in ["iid", "zero_sum"]:
                for l in ["eta1", "eta2"]:
                    singles[(c, l)] = agg(ds, pr, pair, "single", c, l)
            zs_rr = agg(ds, pr, pair, "rr", "zero_sum")
            iid_rr = agg(ds, pr, pair, "rr", "iid")
            fin = {k: v for k, v in singles.items() if v is not None}
            best_key = min(fin, key=fin.get) if fin else None
            best = fin.get(best_key)
            zs_e2 = singles.get(("zero_sum", "eta2"))
            t_zs_rr = agg(ds, pr, pair, "rr", "zero_sum", field="elapsed_sec_per_ensemble")
            t_best = agg(ds, pr, pair, "single", best_key[0], best_key[1], field="elapsed_sec_per_ensemble") if best_key else None
            bias_e1 = agg(ds, pr, pair, "single", "zero_sum", "eta1", field="bias_sq_gold")
            spread = med([r["gold_spread"] for r in est if r["dataset"] == ds and r["prior"] == pr])
            per_cell[(ds, pr)] = dict(best=best, best_key=best_key, zs_rr=zs_rr, iid_rr=iid_rr,
                                      zs_e2=zs_e2, t_zs_rr=t_zs_rr, t_best=t_best,
                                      bias_e1=bias_e1, spread=spread)
            r_best = best / zs_rr if best and zs_rr else None
            r_iid = iid_rr / zs_rr if iid_rr and zs_rr else None
            r_wall = (best * t_best) / (zs_rr * t_zs_rr) if best and zs_rr and t_best and t_zs_rr else None
            print(f"[{ds} prior={pr}] zsRR={fmt(zs_rr)} best_single={fmt(best)}({best_key}) iidRR={fmt(iid_rr)} zs@eta2={fmt(zs_e2)}")
            print(f"    best/zsRR={r_best and round(r_best, 3)}x  iidRR/zsRR={r_iid and round(r_iid, 3)}x  "
                  f"wallclock-adj best/zsRR={r_wall and round(r_wall, 3)}x  "
                  f"bias2(zs@eta1)={fmt(bias_e1)} vs gold_spread={fmt(spread)} "
                  f"ratio={bias_e1 / spread if bias_e1 and spread else None}")
        if pair == "primary":
            ds_pass_strict = {}
            ds_pass_lenient = {}
            never_lose = True
            b2_all = True
            for ds in sorted({c[0] for c in cells}):
                cell_r = []
                for (d, p), v in per_cell.items():
                    if d != ds or v["zs_rr"] is None or v["best"] is None:
                        continue
                    r = v["best"] / v["zs_rr"]
                    cell_r.append(r)
                    if v["zs_rr"] > 1.2 * v["best"]:
                        never_lose = False
                    if v["iid_rr"] is not None and v["iid_rr"] / v["zs_rr"] < 3.0:
                        b2_all = False
                ds_pass_strict[ds] = bool(cell_r) and all(r >= 2.0 for r in cell_r)
                ds_pass_lenient[ds] = bool(cell_r) and any(r >= 2.0 for r in cell_r)
            n_strict = sum(ds_pass_strict.values())
            n_lenient = sum(ds_pass_lenient.values())
            verdict["b1_2x_on_2of3_strict"] = n_strict >= 2
            verdict["b1_2x_on_2of3_lenient"] = n_lenient >= 2
            verdict["b1_never_lose_1.2x"] = never_lose
            verdict["b2_3x_vs_iidRR_all_cells"] = b2_all
            kill2 = any(v["zs_e2"] is not None and v["zs_rr"] is not None and v["zs_e2"] <= v["zs_rr"]
                        for v in per_cell.values())
            verdict["kill2_absorber_zs_eta2_wins_somewhere"] = kill2
            print(f"\nprimary verdicts: {json.dumps(verdict, sort_keys=True)}")
            print(f"  datasets passing 2x strict(both priors)={[d for d, v in ds_pass_strict.items() if v]} "
                  f"lenient(any prior)={[d for d, v in ds_pass_lenient.items() if v]}")


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--selftest-p4", action="store_true",
                        help="CPU-only P4 unit checks (numpy shim; never imports cupy)")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--priors", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--replicates", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--prefix", default="rr_extrapolation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--n-ens", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    # Protocol P4 (gold-only) flags; defaults None/zero = archived behavior.
    parser.add_argument("--gold-only", action="store_true",
                        help="P4: compute only the gold references (frozen etas)")
    parser.add_argument("--start-mode", choices=["zero", "chol", "gauss"], default="zero")
    parser.add_argument("--start-scale", type=float, default=0.0)
    parser.add_argument("--start-seed-base", type=int, default=None)
    parser.add_argument("--dynamics-seed-base", type=int, default=None)
    parser.add_argument("--attempt", type=int, choices=[0, 1], default=0)
    parser.add_argument("--batch-len", type=int, default=1000)
    parser.add_argument("--gold-out", type=Path, default=P4_GOLD_OUT_DEFAULT)
    parser.add_argument("--frozen-gold", type=Path, default=None)
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.selftest_p4:
        sys.exit(selftest_p4())

    global cp
    import cupy as cp_mod
    cp = cp_mod

    if args.evaluate:
        evaluate(args)
        return

    if args.gold_only:
        run_gold_only(args)
        return

    gold_cfg = {"burnin": GOLD_BURNIN, "kept": GOLD_KEPT, "chains": GOLD_CHAINS, "pilot_steps": PILOT_STEPS}
    if args.quick:
        gold_cfg = {"burnin": 150, "kept": 450, "chains": 16, "pilot_steps": 90}

    rep_paths = {rep: args.out_dir / f"{args.prefix}_rep{rep}.jsonl" for rep in args.replicates}
    existing = [str(p) for p in rep_paths.values() if p.exists()]
    if existing:
        sys.exit(f"refusing to append: output file(s) already exist: {existing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold_path = args.out_dir / f"{args.prefix}_gold.jsonl"
    gold_cache = load_gold_cache(gold_path)

    # Gold first (once per cell, shared across replicates).
    data = {}
    golds = {}
    for ds in args.datasets:
        data_x, data_y = load_dataset(args.data_dir, ds)
        data[ds] = (data_x, data_y)
        dim = int(data_x.shape[1])
        for prior in args.priors:
            seeds = [GOLD_SEED_BASE + DS_OFF[ds] + int(round(prior * 1000)) + j for j in range(GOLD_N_SEEDS)]
            gold_id = (f"{ds}|prior={prior}|chains={gold_cfg['chains']}|burnin={gold_cfg['burnin']}"
                       f"|kept={gold_cfg['kept']}|seeds={seeds[0]}-{seeds[-1]}")
            if gold_id in gold_cache:
                golds[(ds, prior)] = gold_cache[gold_id]
                print(f"gold cache hit: {gold_id} spread={gold_cache[gold_id]['gold_spread']:.3e}", flush=True)
                continue
            row = compute_gold(ds, prior, dim, data_x, data_y, args.chunk_rows, gold_cfg)
            golds[(ds, prior)] = row
            with gold_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
            print(f"gold computed: {gold_id} eta={row['mala_eta']:.3f} pilot_accept={row['pilot_accept']:.3f} "
                  f"seed_accepts={[round(a, 3) for a in row['seed_accepts']]} spread={row['gold_spread']:.3e} "
                  f"({row['elapsed_sec']:.1f}s)", flush=True)

    for rep in args.replicates:
        rep_rows = []
        t_rep = time.perf_counter()
        with rep_paths[rep].open("a", encoding="utf-8") as f:
            for ds in args.datasets:
                data_x, data_y = data[ds]
                dim = int(data_x.shape[1])
                cfg = DATASET_CFG[ds]
                n_ens = args.n_ens or (16 if args.quick else cfg["n_ens"])
                budget = args.budget or (120 if args.quick else cfg["budget"])
                for prior in args.priors:
                    gold_row = golds[(ds, prior)]
                    f.write(json.dumps(gold_row, sort_keys=True) + "\n")
                    f.flush()
                    eta_c = default_eta(prior)
                    for est_idx, spec in enumerate(estimator_specs(eta_c, budget)):
                        seed = SEED_BASE + DS_OFF[ds] + int(round(prior * 1000)) + 10000 * rep + 500 * est_idx
                        row = run_estimator(spec, seed, gold_row, n_ens, 8, dim, prior, data_x, data_y, args.chunk_rows)
                        row.update({"dataset": ds, "dim": dim, "prior": prior, "eta_c": eta_c,
                                    "budget": budget, "replicate": rep, "quick": bool(args.quick)})
                        rep_rows.append(row)
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                        f.flush()
                        print(json.dumps({k: row[k] for k in ["dataset", "prior", "pair", "estimator", "coupling",
                                                              "level", "mse_gold_median", "bias_sq_gold", "trace_var",
                                                              "n_finite", "elapsed_sec", "replicate"]}, sort_keys=True), flush=True)
        print_replicate_summary(rep_rows, rep)
        print(f"replicate {rep} wrote {len(rep_rows)} estimator rows to {rep_paths[rep]} "
              f"in {time.perf_counter() - t_rep:.1f}s", flush=True)


if __name__ == "__main__":
    main()
