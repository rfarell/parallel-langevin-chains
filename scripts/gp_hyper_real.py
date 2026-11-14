"""Non-GLM stress cell: GP regression hyperparameters on Superconductivity.

Purpose: one non-GLM cell.  Target: the tempered-likelihood
hyperparameter density exp(-U), U = -log marglik / n - log prior (the
campaign convention exactly as in the logistic cells: likelihood tempered
by 1/n, prior at FULL strength; NOT the untempered posterior, and NOT a
fully tempered one -- tempering the prior too leaves the target nearly
flat in prior directions and puts real mass on numerically singular
kernels), for RBF-kernel GP regression on
an n=320 subset of UCI Superconductivity chosen by a fixed mechanical
permutation (default_rng(320320); index hash recorded) -- a computational
stress cell, not a predictive benchmark.  theta = (log ell, log sf,
log sn), d=3, independent N(0,1) priors on the logs.  The registered
target includes a fixed nugget: K is always K + 1e-6 I (part of the target; routine GP nugget), and
U := +inf on the numerically singular set where float64 Cholesky fails
(exact MALA rejection; NO adaptive jitter).  Solves are validated by an
explicit residual check (max |K K^{-1} - I| < 1e-6 per walker); counters
are per phase, and the zero-singular gate applies to the gold kept phase
and every ULA phase.

Gradients are analytic via dlogML/dtheta_j = tr((alpha alpha^T - K^{-1})
dK/dtheta_j)/2, validated against an independent slow implementation, a
closed-form n=1 case, and central finite differences around zero AND
around the MAP.  Step rule: eta_c = largest eps in 0.4*2^{-k/2} passing
BOTH a MALA pilot (accept >= 0.75, one fixed seed for every grid point)
AND a 600-step iid-ULA stability probe (finite, per-coordinate second
moments < 100).  Gold: MALA, acceptance in [0.55, 0.85], K=64, 5000
burn-in, 30000 kept, 3 seeds pooled, campaign-convention spread (max
pairwise squared seed-mean distance), per-seed half means recorded.
Seeds 32x,xxx,xxx block (verified unused).  JSONL sort_keys append,
refuses overwrite, full provenance per row.
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

try:
    import cupy as cp
except ImportError:  # CPU fallback (selftest/MAP paths)
    cp = None

METHODS = ["iid", "zero_sum", "antithetic"]
N_SUB = 320
SUBSET_PERM_SEED = 320320
EPS_SCAN = [0.4 * 2.0 ** (-k / 2.0) for k in range(16)]
REP_SEED_BASE = 320_000_000
GOLD_SEED = 321_000_000
PILOT_SEED = 322_000_000
INV_SEED = 323_000_000
CHUNK_WALKERS = 64
JITTER = 1e-6
RESID_TOL = 1e-6
CANON = dict(chains=8, reps=24, steps=2400, burnin=1200,
             gold_chains=64, gold_kept=30000, gold_burnin=5000)


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


def row_sha256(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:16]


def provenance(args):
    import scipy
    return {
        "argv": " ".join(sys.argv),
        "script_sha256": file_sha256(__file__),
        "precommit_sha256": file_sha256("campaigns/gp_hyperposterior_precommit.md"),
        "data_sha256": file_sha256(args.data_dir / "superconduct.npz"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cupy_version": cp.__version__ if cp is not None else "none",
    }


def load_gp_data(data_dir):
    data = np.load(data_dir / "superconduct.npz", allow_pickle=True)
    x_all = np.asarray(data["X_train"], dtype=np.float64)
    y_all = np.asarray(data["y_train"], dtype=np.float64)
    idx = np.random.default_rng(SUBSET_PERM_SEED).permutation(len(y_all))[:N_SUB]
    idx_hash = hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()[:16]
    x = x_all[idx]
    y = y_all[idx]
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)
    y_mean, y_std = float(y.mean()), float(y.std())
    y = (y - y_mean) / y_std
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    return x, y, d2, y_mean, y_std, idx_hash


class GPPotential:
    """U(theta) and grad for a batch of walkers, float64, GPU or CPU.

    Registered target: K := K_rbf + JITTER*I; U = +inf where float64
    Cholesky of K fails or the solve residual exceeds RESID_TOL (exact
    MALA rejection).  Counters accumulate per phase (set_phase)."""

    def __init__(self, d2, y, xp):
        self.xp = xp
        self.d2 = xp.asarray(d2)
        self.y = xp.asarray(y)
        self.n = y.shape[0]
        self.eye = xp.eye(self.n, dtype=self.d2.dtype)
        self.phase = "init"
        self.singular = {}
        self.max_resid = {}

    def set_phase(self, name):
        self.phase = name
        self.singular.setdefault(name, 0)
        self.max_resid.setdefault(name, 0.0)

    def counters(self):
        return {"singular_by_phase": dict(self.singular),
                "max_solve_resid_by_phase": {k: float(v) for k, v in self.max_resid.items()}}

    def _chol_batch(self, kj):
        xp = self.xp
        if xp is np:
            out = np.full_like(kj, np.nan)
            for i in range(kj.shape[0]):
                try:
                    out[i] = np.linalg.cholesky(kj[i])
                except np.linalg.LinAlgError:
                    pass
            return out
        return xp.linalg.cholesky(kj)

    def u_grad(self, theta):
        """theta: (W,3) -> U (W,), grad (W,3); U=+inf on invalid walkers."""
        xp = self.xp
        w_n = theta.shape[0]
        u_out = xp.empty(w_n, dtype=xp.float64)
        g_out = xp.empty((w_n, 3), dtype=xp.float64)
        self.singular.setdefault(self.phase, 0)
        self.max_resid.setdefault(self.phase, 0.0)
        for lo in range(0, w_n, CHUNK_WALKERS):
            hi = min(lo + CHUNK_WALKERS, w_n)
            th = theta[lo:hi]
            ell2 = xp.exp(2.0 * th[:, 0])[:, None, None]
            sf2 = xp.exp(2.0 * th[:, 1])[:, None, None]
            sn2 = xp.exp(2.0 * th[:, 2])[:, None, None]
            e_mat = xp.exp(-self.d2[None] / (2.0 * ell2))
            k_mat = sf2 * e_mat + (sn2 + JITTER) * self.eye[None]
            finite_row = xp.isfinite(k_mat).all(axis=(1, 2))
            k_safe = xp.where(finite_row[:, None, None], k_mat, self.eye[None])
            l_mat = self._chol_batch(k_safe)
            good = finite_row & ~xp.isnan(l_mat).any(axis=(1, 2))
            k_solve = xp.where(good[:, None, None], k_safe, self.eye[None])
            kinv = xp.linalg.solve(k_solve, xp.broadcast_to(self.eye[None], k_solve.shape).copy())
            resid = xp.abs(k_solve @ kinv - self.eye[None]).max(axis=(1, 2))
            good = good & xp.isfinite(kinv).all(axis=(1, 2)) & (resid < RESID_TOL)
            self.singular[self.phase] += int((~good).sum())
            self.max_resid[self.phase] = max(self.max_resid[self.phase],
                                             float(xp.where(good, resid, 0.0).max()))
            alpha = kinv @ self.y
            diag_l = xp.where(good[:, None], xp.diagonal(l_mat, axis1=1, axis2=2), 1.0)
            logdet = 2.0 * xp.log(diag_l).sum(axis=1)
            quad = alpha @ self.y
            logml = -0.5 * quad - 0.5 * logdet - 0.5 * self.n * math.log(2.0 * math.pi)
            logprior = -0.5 * (th * th).sum(axis=1) - 1.5 * math.log(2.0 * math.pi)
            # Campaign convention (matches the logistic cells): likelihood
            # tempered by 1/n, prior at FULL strength.
            u_val = -logml / self.n - logprior
            aat_minus = alpha[:, :, None] * alpha[:, None, :] - kinv
            dk_ell = sf2 * e_mat * (self.d2[None] / ell2)
            dk_sf = 2.0 * sf2 * e_mat
            g_ell = 0.5 * (aat_minus * dk_ell).sum(axis=(1, 2))
            g_sf = 0.5 * (aat_minus * dk_sf).sum(axis=(1, 2))
            g_sn = 0.5 * (2.0 * sn2[:, 0, 0]) * xp.diagonal(aat_minus, axis1=1, axis2=2).sum(axis=1)
            dlogml = xp.stack([g_ell, g_sf, g_sn], axis=1)
            g_val = -dlogml / self.n + th
            u_out[lo:hi] = xp.where(good, u_val, xp.inf)
            g_out[lo:hi] = xp.where(good[:, None], g_val, 0.0)
        return u_out, g_out


def make_noise(shape, method, xp):
    reps, chains, _ = shape
    def snormal(shp):
        if xp is np:
            return np.random.standard_normal(shp)
        return xp.random.standard_normal(shp, dtype=xp.float64)
    if method == "iid":
        return snormal(shape)
    if method == "zero_sum":
        z = snormal(shape)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
    if method == "antithetic":
        if chains % 2:
            z = snormal((reps, chains // 2, shape[-1]))
            extra = snormal((reps, 1, shape[-1]))
            return xp.concatenate([z, -z, extra], axis=1)
        z = snormal((reps, chains // 2, shape[-1]))
        return xp.concatenate([z, -z], axis=1)
    raise ValueError(method)


def to_np(xp, arr):
    return cp.asnumpy(arr) if (cp is not None and xp is cp) else np.asarray(arr)


def mala_run(pot, xp, seed, eps, chains, steps, burnin, start):
    xp.random.seed(seed)
    x = xp.tile(xp.asarray(start, dtype=xp.float64)[None], (chains, 1))
    u, g = pot.u_grad(x)
    acc_mean = xp.zeros(3, dtype=xp.float64)
    acc2 = xp.zeros(3, dtype=xp.float64)
    acc3 = xp.zeros(3, dtype=xp.float64)
    half1 = xp.zeros(3, dtype=xp.float64)
    accept_acc = xp.zeros((), dtype=xp.float64)
    kept = 0
    kept_half = (steps - burnin) // 2
    for step in range(steps):
        noise = (np.random.standard_normal(x.shape) if xp is np
                 else xp.random.standard_normal(x.shape, dtype=xp.float64))
        prop = x - eps * g + math.sqrt(2.0 * eps) * noise
        u_p, g_p = pot.u_grad(prop)
        fwd = ((prop - x + eps * g) ** 2).sum(axis=1)
        bwd = ((x - prop + eps * g_p) ** 2).sum(axis=1)
        log_alpha = (u - u_p) + (fwd - bwd) / (4.0 * eps)
        accept = xp.log(xp.random.random(x.shape[0])) < log_alpha
        accept_acc += accept.mean()
        acc_n = accept[:, None]
        x = xp.where(acc_n, prop, x)
        u = xp.where(accept, u_p, u)
        g = xp.where(acc_n, g_p, g)
        if step >= burnin:
            acc_mean += x.mean(axis=0)
            acc2 += (x * x).mean(axis=0)
            acc3 += (x ** 3).mean(axis=0)
            if kept < kept_half:
                half1 += x.mean(axis=0)
            kept += 1
    rate = float(to_np(xp, accept_acc)) / steps
    mean = acc_mean / kept
    half1_mean = half1 / kept_half
    half2_mean = (acc_mean - half1) / (kept - kept_half)
    return mean, acc2 / kept, acc3 / kept, rate, half1_mean, half2_mean


CLIP_G = 10.0


def clipped_drift(g, xp):
    """Standard stiffness guard: per-walker gradient-norm clip at CLIP_G.
    Deterministic per-chain, identical across couplings, never touches the
    noise -- marginal invariance and coupling comparisons remain exact.
    Returns (clipped gradient, number of clipped walkers)."""
    gnorm = xp.sqrt((g * g).sum(axis=-1, keepdims=True))
    scale = xp.minimum(1.0, CLIP_G / xp.maximum(gnorm, 1e-300))
    return g * scale, int((scale < 1.0).sum())


def ula_probe(pot, xp, seed, eps, reps, chains, steps, method="iid"):
    """Runs coupled clipped ULA; returns per-coordinate chain-0 second
    moments per ensemble (reps, 3), a finite flag, and clip activations."""
    xp.random.seed(seed)
    x = xp.zeros((reps, chains, 3), dtype=xp.float64)
    acc = xp.zeros((reps, 3), dtype=xp.float64)
    kept = 0
    clips = 0
    for step in range(steps):
        flat = x.reshape(-1, 3)
        _, g = pot.u_grad(flat)
        g, n_clip = clipped_drift(g, xp)
        clips += n_clip
        x = x - eps * g.reshape(x.shape) + math.sqrt(2.0 * eps) * make_noise(x.shape, method, xp)
        if step >= steps // 2:
            acc += x[:, 0, :] ** 2
            kept += 1
    m2 = to_np(xp, acc / kept)
    return m2, bool(np.isfinite(m2).all()), clips


def calibrate_eta(pot, xp):
    """ULA step: largest eps passing the MALA-acceptance pilot (>= 0.75),
    the iid-ULA stability probe, AND a singular-free probe (zero
    singular-K events, so the zero-singular ULA gate is achievable by
    construction).  Gold step (separate, per pre-commit): largest eps
    whose MALA pilot acceptance lies in [0.55, 0.85].  One fixed seed
    reused for every grid point."""
    rates, stable = {}, {}
    eta_ula, eta_gold = None, None
    for eps in EPS_SCAN:
        pot.set_phase("pilot")
        _, _, _, rate, _, _ = mala_run(pot, xp, PILOT_SEED, eps, 8, 200, 100, np.zeros(3))
        probe_phase = f"probe_{eps:.6g}"
        pot.set_phase(probe_phase)
        m2, finite, _ = ula_probe(pot, xp, PILOT_SEED + 1, eps, 8, 8, 600)
        ok_stable = (finite and float(m2.max()) < 100.0
                     and pot.singular[probe_phase] == 0)
        rates[f"{eps:.6g}"] = rate
        stable[f"{eps:.6g}"] = ok_stable
        if eta_gold is None and 0.55 <= rate <= 0.85:
            eta_gold = eps
        if eta_ula is None and rate >= 0.75 and ok_stable:
            eta_ula = eps
    if eta_ula is None or eta_gold is None:
        print(json.dumps({"kind": "pilot_failure", "rates": rates, "stable": stable},
                         sort_keys=True), flush=True)
        raise SystemExit("pilot found no eps passing the gates")
    print(json.dumps({"calibrate": "gp_hyper", "eta_ula": eta_ula, "eta_gold": eta_gold,
                      "rates": rates, "stable": stable}, sort_keys=True), flush=True)
    return eta_ula, eta_gold, rates, stable


def marginal_invariance_pilot(pot, xp, eta):
    """4-sigma across-ensemble gate for zero_sum AND antithetic vs iid."""
    pot.set_phase("invariance")
    reps = 64
    m2 = {}
    for i, method in enumerate(METHODS):
        m2[method], finite, _ = ula_probe(pot, xp, INV_SEED + i, eta, reps, 8, 600, method)
        if not finite:
            raise SystemExit(f"invariance pilot: non-finite states under {method}")
    worst = 0.0
    base = m2["iid"]
    se_base = base.std(axis=0, ddof=1) / math.sqrt(reps)
    for method in ("zero_sum", "antithetic"):
        se_m = m2[method].std(axis=0, ddof=1) / math.sqrt(reps)
        z = np.abs(m2[method].mean(axis=0) - base.mean(axis=0)) / np.sqrt(se_base ** 2 + se_m ** 2)
        worst = max(worst, float(z.max()))
    print(f"marginal invariance pilot: max |z| {worst:.2f} (gate 4)")
    if worst > 4.0:
        raise SystemExit("marginal invariance pilot FAILED")
    if pot.singular["invariance"] > 0:
        raise SystemExit(f"singular events during invariance pilot: {pot.singular['invariance']}")


def gold_reference(pot, xp, eta, seeds, kept_steps, burnin, chains):
    pot.set_phase("gold")
    per_seed, accepts, m2_all, m3_all, halves = [], [], [], [], []
    for seed in seeds:
        mean, acc2, acc3, rate, h1, h2 = mala_run(pot, xp, seed, eta, chains,
                                                  burnin + kept_steps, burnin, np.zeros(3))
        per_seed.append(to_np(xp, mean))
        accepts.append(rate)
        m2_all.append(to_np(xp, acc2))
        m3_all.append(to_np(xp, acc3))
        halves.append([to_np(xp, h1).tolist(), to_np(xp, h2).tolist()])
    pooled = np.mean(per_seed, axis=0)
    # campaign convention: maximum pairwise squared distance between seeds
    spread = float(max(np.sum((a - b) ** 2)
                       for i, a in enumerate(per_seed) for b in per_seed[i + 1:]))
    m2 = np.mean(m2_all, axis=0)
    m3 = np.mean(m3_all, axis=0)
    var = m2 - pooled ** 2
    skew = (m3 - 3 * pooled * var - pooled ** 3) / np.maximum(var, 1e-30) ** 1.5
    return pooled, spread, per_seed, accepts, var, skew, halves


def ula_method(pot, xp, design, method, ref):
    rng_seed = design["seed"] + 101 * (METHODS.index(method) + 1) + 60013
    xp.random.seed(rng_seed)
    pot.set_phase(f"ula_{method}")
    reps, chains = design["reps"], design["chains"]
    x = xp.zeros((reps, chains, 3), dtype=xp.float64)
    eta = design["eta"]
    ens_acc = xp.zeros((reps, 3), dtype=xp.float64)
    windows = 0
    start = time.perf_counter()
    clips = 0
    for step in range(design["steps"]):
        flat = x.reshape(-1, 3)
        _, g = pot.u_grad(flat)
        g, n_clip = clipped_drift(g, xp)
        clips += n_clip
        x = x - eta * g.reshape(x.shape) + math.sqrt(2.0 * eta) * make_noise(x.shape, method, xp)
        if step >= design["burnin"]:
            ens_acc += x.mean(axis=1)
            windows += 1
    elapsed = time.perf_counter() - start
    ens = to_np(xp, ens_acc) / windows
    if not np.isfinite(ens).all():
        raise SystemExit(f"non-finite ULA estimates under {method}")
    if pot.singular[f"ula_{method}"] > 0:
        raise SystemExit(f"singular events during ULA ({method}): "
                         f"{pot.singular[f'ula_{method}']}")
    err = ens - ref[None, :]
    centered = ens - ens.mean(axis=0)[None, :]
    return {
        **design,
        "target": "gp_hyper_superconduct_tempered",
        "kind": "ula",
        "method": method,
        "coupling": method,
        "actual_rng_seed": rng_seed,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "windows": windows,
        "mse_gold": float(np.mean(np.sum(err * err, axis=1))),
        "mse_gold_median": float(np.median(np.sum(err * err, axis=1))),
        "trace_var": float(np.mean(np.sum(centered * centered, axis=1))),
        "bias_sq": float(np.sum((ens.mean(axis=0) - ref) ** 2)),
        "kernel": "clipped_ula",
        "clip_g": CLIP_G,
        "clip_activations": clips,
        "clip_rate": clips / (design["steps"] * design["reps"] * design["chains"]),
        **pot.counters(),
    }


def map_baseline(pot_np):
    from scipy.optimize import minimize

    pot_np.set_phase("map")

    def f(th):
        u, g = pot_np.u_grad(th[None])
        return float(u[0]), np.asarray(g[0])

    res = minimize(f, np.zeros(3), jac=True, method="BFGS",
                   options=dict(gtol=1e-12, maxiter=2000))
    gnorm = float(np.max(np.abs(res.jac)))
    if not res.success and gnorm > 1e-8:
        raise SystemExit(f"MAP did not converge: {res.message}")
    return np.asarray(res.x), gnorm


def selftest(data_dir):
    x, y, d2, _, _, idx_hash = load_gp_data(data_dir)
    pot = GPPotential(d2, y, np)
    pot.set_phase("selftest")
    rng = np.random.default_rng(1390001)
    # independent slow reference for U at a few points (includes the jitter,
    # which is part of the registered target)
    for _ in range(3):
        th = rng.standard_normal(3) * 0.5
        ell2, sf2, sn2 = np.exp(2 * th[0]), np.exp(2 * th[1]), np.exp(2 * th[2])
        k_ref = sf2 * np.exp(-d2 / (2 * ell2)) + (sn2 + JITTER) * np.eye(len(y))
        sign, logdet = np.linalg.slogdet(k_ref)
        assert sign > 0
        alpha = np.linalg.solve(k_ref, y)
        logml = -0.5 * y @ alpha - 0.5 * logdet - 0.5 * len(y) * math.log(2 * math.pi)
        logprior = -0.5 * th @ th - 1.5 * math.log(2 * math.pi)
        u_ref = -logml / len(y) - logprior
        u, _ = pot.u_grad(th[None])
        rel = abs(float(u[0]) - u_ref) / abs(u_ref)
        assert rel < 1e-12, f"U mismatch: {rel}"
    # closed-form n=1 sanity: K = sf^2 + sn^2 + jitter (scalar)
    y1 = np.array([0.7])
    pot1 = GPPotential(np.zeros((1, 1)), y1, np)
    pot1.set_phase("selftest")
    th = np.array([0.3, -0.2, 0.1])
    kval = math.exp(2 * th[1]) + math.exp(2 * th[2]) + JITTER
    logml1 = -0.5 * y1[0] ** 2 / kval - 0.5 * math.log(kval) - 0.5 * math.log(2 * math.pi)
    logprior1 = -0.5 * th @ th - 1.5 * math.log(2 * math.pi)
    u1_ref = -logml1 / 1.0 - logprior1
    u1, _ = pot1.u_grad(th[None])
    rel1 = abs(float(u1[0]) - u1_ref) / abs(u1_ref)
    assert rel1 < 1e-12, f"n=1 closed form mismatch: {rel1}"
    # FD gradient around zero AND around the MAP
    theta_map, gnorm = map_baseline(GPPotential(d2, y, np))
    worst = 0.0
    for i in range(20):
        base = theta_map if i % 2 else np.zeros(3)
        th = base + rng.standard_normal(3) * 0.2
        _, g = pot.u_grad(th[None])
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1e-6
            up, _ = pot.u_grad((th + e)[None])
            um, _ = pot.u_grad((th - e)[None])
            fd = (float(up[0]) - float(um[0])) / 2e-6
            rel = abs(fd - float(g[0, j])) / max(abs(fd), abs(float(g[0, j])), 1e-12)
            worst = max(worst, rel)
    print(f"selftest: U matches slow reference and n=1 closed form; "
          f"MAP gnorm {gnorm:.1e}; grad FD worst rel {worst:.3e}; subset {idx_hash}")
    assert worst < 1e-6, "gradient FD check failed"
    print("selftest PASS")


BANDS = [(0.5, "transfers"), (0.8, "partial"), (1.25, "parity"), (np.inf, "hurts")]


def band_of(r):
    for hi, name in BANDS:
        if (r <= 0.5 and name == "transfers") or \
           (0.5 < r <= 0.8 and name == "partial") or \
           (0.8 < r < 1.25 and name == "parity") or \
           (r >= 1.25 and name == "hurts"):
            return name
    return "?"


def fmt(x, sig=3):
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        mant, exp = f"{x:.{sig - 1}e}".split("e")
        return f"${mant}\\times 10^{{{int(exp)}}}$"
    decimals = max(sig - 1 - int(np.floor(np.log10(abs(x)))), 0)
    return f"{x:.{decimals}f}"


def cell_fmt(vals, sig=3):
    med = float(np.median(vals))
    return f"{fmt(med, sig)} [{fmt(min(vals), sig)}, {fmt(max(vals), sig)}]"


def load_gold_row(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "gold":
                return row
    raise SystemExit(f"no gold row in {path}")


def make_table(table_inputs, table_out):
    rows = []
    for p in table_inputs:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    ula = [r for r in rows if r.get("kind") == "ula"]
    gold = next(r for r in rows if r.get("kind") == "gold")
    map_row = next(r for r in rows if r.get("kind") == "map_baseline")
    gold_sha = row_sha256(gold)
    # canonical validator (pre-registered)
    assert not gold.get("quick") and gold["chains"] == CANON["gold_chains"] \
        and gold["kept"] == CANON["gold_kept"] and gold["burnin"] == CANON["gold_burnin"], \
        "gold row is not the canonical configuration"
    reps_seen = sorted({r["replicate"] for r in ula})
    assert reps_seen == [0, 1, 2], f"replicates {reps_seen}"
    for rep in reps_seen:
        for m in METHODS:
            sub = [r for r in ula if r["replicate"] == rep and r["method"] == m]
            assert len(sub) == 1, f"expected 1 row for rep {rep} {m}, got {len(sub)}"
            r = sub[0]
            assert not r.get("quick"), "quick ULA row in table inputs"
            for field, want in (("chains", CANON["chains"]), ("reps", CANON["reps"]),
                                ("steps", CANON["steps"]), ("burnin", CANON["burnin"])):
                assert r[field] == want, f"{field}={r[field]} != canonical {want}"
            assert r["eta"] == gold["eta"], "eta mismatch between ULA row and gold"
            assert r["gold_id"] == gold_sha, "gold_id does not match the gold row sha"
            assert np.isfinite(r["trace_var"]) and np.isfinite(r["mse_gold_median"])
    ratios = {m: [] for m in ("zero_sum", "antithetic")}
    mses = {m: [] for m in METHODS}
    for rep in reps_seen:
        by_m = {r["method"]: r for r in ula if r["replicate"] == rep}
        for m in ("zero_sum", "antithetic"):
            ratios[m].append(by_m[m]["trace_var"] / by_m["iid"]["trace_var"])
        for m in METHODS:
            mses[m].append(by_m[m]["mse_gold_median"])
    # pre-registered kill criterion
    smallest = min(min(v) for v in mses.values())
    assert gold["gold_spread"] < 0.1 * smallest, (
        f"KILL: gold spread {gold['gold_spread']:.3g} not < 10% of smallest "
        f"tabulated error {smallest:.3g}; cell unresolved")
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        " & iid & $C_\\star$ & pairwise \\\\",
        "\\midrule",
        ("trace variance / iid & 1 & "
         f"{cell_fmt(ratios['zero_sum'])} & {cell_fmt(ratios['antithetic'])} \\\\"),
        ("median $\\|\\widehat\\theta-\\theta_{\\mathrm{gold}}\\|^2$ & "
         f"{cell_fmt(mses['iid'])} & {cell_fmt(mses['zero_sum'])} & "
         f"{cell_fmt(mses['antithetic'])} \\\\"),
        "\\midrule",
        (f"MAP $\\|\\theta-\\theta_{{\\mathrm{{gold}}}}\\|^2$: {fmt(map_row['map_err_sq_gold'])}"
         f" & \\multicolumn{{3}}{{r}}{{gold spread {gold['gold_spread']:.1e};"
         f" gold skewness {', '.join(f'{s:.2f}' for s in gold['gold_skew'])}}} \\\\"),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    Path(table_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {table_out}")
    for m in ("zero_sum", "antithetic"):
        med = float(np.median(ratios[m]))
        bands_hit = {band_of(v) for v in ratios[m]}
        note = "" if len(bands_hit) == 1 else " (replicate range straddles band boundary: unresolved at this replication)"
        print(f"  {m}/iid median {med:.3g} -> band '{band_of(med)}'{note}; reps {sorted(ratios[m])}")
    # ordering-language rule for the end-task line (1.5x + five gold spreads)
    for a, b in (("zero_sum", "iid"), ("zero_sum", "antithetic")):
        ma, mb = float(np.median(mses[a])), float(np.median(mses[b]))
        sep = max(ma, mb) / max(min(ma, mb), 1e-300)
        both_resolved = min(ma, mb) > 5 * gold["gold_spread"]
        ordered = sep >= 1.5 and both_resolved
        print(f"  mse {a} vs {b}: {ma:.3g} vs {mb:.3g} -> "
              f"{'ordered' if ordered else 'comparable at reference precision'}")
    print(f"  MAP {map_row['map_err_sq_gold']:.3g}; spread {gold['gold_spread']:.1e}; "
          f"skew {gold['gold_skew']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gold", "ula"], default="ula")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--gold", type=Path, default=Path("outputs/sweeps/gp_hyper_gold.jsonl"))
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--chains", type=int, default=CANON["chains"])
    parser.add_argument("--reps", type=int, default=CANON["reps"])
    parser.add_argument("--steps", type=int, default=CANON["steps"])
    parser.add_argument("--burnin", type=int, default=CANON["burnin"])
    parser.add_argument("--gold-kept", type=int, default=CANON["gold_kept"])
    parser.add_argument("--gold-burnin", type=int, default=CANON["gold_burnin"])
    parser.add_argument("--gold-chains", type=int, default=CANON["gold_chains"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--make-table", action="store_true")
    parser.add_argument("--table-inputs", nargs="+", type=Path,
                        default=[Path("outputs/sweeps/gp_hyper_gold.jsonl")]
                        + [Path(f"outputs/sweeps/gp_hyper_rep{r}.jsonl") for r in range(3)])
    parser.add_argument("--table-out", type=Path, default=Path("paper/tables/gp_hyper.tex"))
    args = parser.parse_args()
    if args.selftest:
        selftest(args.data_dir)
        return
    if args.make_table:
        make_table(args.table_inputs, args.table_out)
        return
    if args.out is None:
        parser.error("--out required")
    if args.out.exists():
        parser.error(f"refusing to overwrite existing {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.reps = 8
        args.steps = 300
        args.burnin = 150
        args.gold_kept = 2000
        args.gold_burnin = 500
        args.gold_chains = 16

    xp = cp if cp is not None else np
    x, y, d2, y_mean, y_std, idx_hash = load_gp_data(args.data_dir)
    pot = GPPotential(d2, y, xp)
    prov = provenance(args)

    if args.mode == "gold":
        pot_np = GPPotential(d2, y, np)
        eta_ula, eta_gold, pilot_rates, pilot_stable = calibrate_eta(pot, xp)
        marginal_invariance_pilot(pot, xp, eta_ula)
        theta_map, map_gnorm = map_baseline(pot_np)
        gold_seeds = [GOLD_SEED + 11 * s for s in range(3)]
        pooled, spread, per_seed, accepts, gold_var, gold_skew, halves = gold_reference(
            pot, xp, eta_gold, gold_seeds, args.gold_kept, args.gold_burnin, args.gold_chains)
        if not all(0.5 <= a <= 0.9 for a in accepts):
            raise SystemExit(f"gold acceptance out of [0.5, 0.9]: {accepts}")
        if not np.isfinite(pooled).all():
            raise SystemExit("gold mean not finite")
        if pot.singular["gold"] > 0:
            raise SystemExit(f"singular events during gold: {pot.singular['gold']}")
        gold_row = {
            "kind": "gold",
            "target": "gp_hyper_superconduct_tempered",
            "n_sub": N_SUB,
            "subset_perm_seed": SUBSET_PERM_SEED,
            "subset_idx_sha256": idx_hash,
            "eta": eta_ula,
            "eta_gold": eta_gold,
            "quick": bool(args.quick),
            "pilot_rates": pilot_rates,
            "pilot_stable": pilot_stable,
            "gold_mean": [float(v) for v in pooled],
            "gold_spread": spread,
            "gold_var": [float(v) for v in gold_var],
            "gold_skew": [float(v) for v in gold_skew],
            "seed_means": [[float(v) for v in m] for m in per_seed],
            "seed_half_means": halves,
            "seed_accepts": accepts,
            "gold_seeds": [GOLD_SEED + 11 * s for s in range(3)],
            "chains": args.gold_chains,
            "kept": args.gold_kept,
            "burnin": args.gold_burnin,
            "y_standardization": [y_mean, y_std],
            **pot.counters(),
            "provenance": prov,
            "git_hash": git_hash(),
        }
        rows = [gold_row, {
            "kind": "map_baseline",
            "target": "gp_hyper_superconduct_tempered",
            "theta_map": [float(v) for v in theta_map],
            "map_gnorm": map_gnorm,
            "map_err_sq_gold": float(np.sum((theta_map - pooled) ** 2)),
            "gold_id": row_sha256(gold_row),
            "provenance": prov,
            "git_hash": git_hash(),
        }]
    else:
        g = load_gold_row(args.gold)
        pooled = np.asarray(g["gold_mean"], dtype=np.float64)
        design = {
            "dataset": "superconduct_gp",
            "dim": 3,
            "chains": args.chains,
            "eta": g["eta"],
            "gold_id": row_sha256(g),
            "quick": bool(args.quick),
            "steps": args.steps,
            "burnin": args.burnin,
            "reps": args.reps,
            "replicate": args.replicate,
            "seed": REP_SEED_BASE + 100_000 * args.replicate,
        }
        rows = []
        for method in METHODS:
            row = ula_method(pot, xp, design, method, pooled)
            row["provenance"] = prov
            rows.append(row)
            print(json.dumps({k: row[k] for k in
                              ("method", "trace_var", "mse_gold_median", "bias_sq")},
                             sort_keys=True), flush=True)

    with args.out.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
