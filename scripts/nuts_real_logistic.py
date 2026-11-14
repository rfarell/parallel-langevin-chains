"""Adaptive path lengths and NUTS under coupled momentum refresh.

Purpose: complete the kernel ladder (ULA -> MALA -> HMC -> adaptive).
Three kernels on the real logistic cells, coupled ONLY in the momentum
refresh (direction bits and in-trajectory selection stay per-chain iid):

  jitter_shared:   HMC with L ~ Uniform{1..2*Lbar} drawn per outer step
                   PER ENSEMBLE, shared by that ensemble's K chains only
                   (parallel-chain adaptation schemes like ChEES tune ONE
                   trajectory length for all chains); across-ensemble
                   variance therefore marginalizes over schedules for both
                   jitter kernels.  Mixture of exact HMC kernels =>
                   pi-invariant.
  jitter_perchain: same law per chain, but each chain draws its own L
                   (proxy for per-chain adaptation).  Chain-marginally
                   identical to jitter_shared; only cross-chain alignment
                   of the integration map differs.
  nuts:            multinomial NUTS (biased progressive sampling across
                   doublings, uniform progressive within subtrees) with the
                   original endpoint U-turn criterion checked on every
                   aligned power-of-two block via iterative checkpointing
                   (state at even in-subtree index k stored in slot
                   popcount(k); a block of size 2^m completes at index k
                   with m-th trailing one bit, its left endpoint sits in
                   slot popcount(k)-m).

The L-jitter sequences come from a design-seeded RNG shared by all
couplings within a design (like the step size), so coupling ratios are not
confounded by different length draws.  Metric mirrors
scripts/hmc_real_logistic.py: matched posterior-mean trace-variance ratios
against a kernel-matched long iid reference at a per-cell pilot-calibrated
eps (same pilot seeds as the HMC campaign => same eps).  Seed block
31x,xxx,xxx (tuple-based over dataset/replicate/design), verified unused
by any prior campaign; energies accumulate in float64; every row records
provenance (argv, script/precommit sha256, versions, quick flag).
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

import cupy as cp
import numpy as np


METHODS = ["iid", "zero_sum", "antithetic"]
KERNELS = ["jitter_shared", "jitter_perchain", "nuts"]
DATASETS = ["wdbc", "spambase", "adult"]
DATASET_DEFAULTS = {
    "wdbc": {"steps": 300, "burnin": 150, "ref_steps": 600, "ref_burnin": 300, "reps": 128},
    "spambase": {"steps": 300, "burnin": 150, "ref_steps": 600, "ref_burnin": 300, "reps": 96},
    "adult": {"steps": 300, "burnin": 150, "ref_steps": 600, "ref_burnin": 300, "reps": 48},
}
LBAR = 10
MAX_DEPTH = 8
DIVERGENCE = 1000.0
EPS_SCAN = [6.4 * 2.0 ** (-k / 2.0) for k in range(18)]
DATASET_LABELS = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}
KERNEL_LABELS = {"jitter_shared": "jitter (shared $L$)",
                 "jitter_perchain": "jitter (per-chain $L$)",
                 "nuts": "NUTS"}


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
    import scipy
    return {
        "argv": " ".join(sys.argv),
        "script_sha256": file_sha256(__file__),
        "precommit_sha256": file_sha256("campaigns/nuts_adaptive_precommit.md"),
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__,
        "scipy_version": scipy.__version__,
    }


def dot64(a, b):
    return (a.astype(cp.float64) * b.astype(cp.float64)).sum(axis=-1)


def make_noise(shape, method, dtype=cp.float32):
    reps, chains, _ = shape
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        z = cp.random.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)
        return z * math.sqrt(chains / (chains - 1))
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


def logistic_potential(data_x, data_y, prior, chunk_rows):
    n = data_x.shape[0]

    def potential(flat):
        grad_out = cp.empty_like(flat)
        u_out = cp.empty(flat.shape[0], dtype=cp.float32)
        prior_mask = cp.ones(flat.shape[-1], dtype=cp.float32)
        prior_mask[0] = 0.0
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
        return u_out, grad_out

    return potential


def gaussian_potential(lams):
    lams_cp = cp.asarray(lams, dtype=cp.float32)

    def potential(flat):
        u = 0.5 * cp.sum(lams_cp[None, :] * flat * flat, axis=1)
        grad = lams_cp[None, :] * flat
        return u, grad

    return potential


def dot(a, b):
    return cp.sum(a * b, axis=-1)


# ---------------------------------------------------------------------------
# Jittered-length HMC: exact leapfrog(L_c) per chain, vectorized over walkers
# with per-walker L; MH-corrected; diag captures the rejection-leak factors.
# ---------------------------------------------------------------------------

def jitter_leapfrog(x, g, p0, eps, l_steps, potential):
    """Exact per-walker leapfrog(l_steps[w]); returns (xf, uf, gf, pf)."""
    l_max = int(l_steps.max())
    ph = p0 - 0.5 * eps * g
    xt = x.copy()
    xf = x.copy()
    uf = cp.zeros(x.shape[0], dtype=cp.float32)
    gf = g.copy()
    pf = cp.zeros_like(p0)
    for l in range(1, l_max + 1):
        move = (l_steps >= l)
        xt = cp.where(move[:, None], xt + eps * ph, xt)
        ut, gt = potential(xt)
        fin = (l_steps == l)
        pf = cp.where(fin[:, None], ph - 0.5 * eps * gt, pf)
        xf = cp.where(fin[:, None], xt, xf)
        uf = cp.where(fin, ut, uf)
        gf = cp.where(fin[:, None], gt, gf)
        cont = (l_steps > l)
        ph = cp.where(cont[:, None], ph - eps * gt, ph)
    return xf, uf, gf, pf


def jitter_propose(x, u, g, p0, eps, l_steps, potential):
    """Propose half of jitter_hmc_step (no RNG, no state update).

    eps may be a scalar or a per-walker (W, 1) float32 array (per-ensemble
    step sizes broadcast to the walker axis); jitter_leapfrog broadcasts
    either form identically.  Returns (xf, uf, gf, pf, log_alpha, alpha)
    with log_alpha the totalized float64 acceptance threshold
    min(0, H0 - H) if H0 - H is finite else -inf, and alpha = exp(log_alpha)
    (nonfinite proposals reject with alpha = 0; no NaN enters any weight)."""
    h_old = u.astype(cp.float64) + 0.5 * dot64(p0, p0)
    xf, uf, gf, pf = jitter_leapfrog(x, g, p0, eps, l_steps, potential)
    h_new = uf.astype(cp.float64) + 0.5 * dot64(pf, pf)
    dh = h_old - h_new
    log_alpha = cp.where(cp.isfinite(dh), cp.minimum(dh, 0.0), -cp.inf)
    return xf, uf, gf, pf, log_alpha, cp.exp(log_alpha)


def jitter_accept(x, u, g, xf, uf, gf, log_alpha, accept_acc, p0=None, diag=None,
                  u_unif=None):
    """Accept half: draws the per-walker float32 uniform (u_unif injection is
    for SELFTESTS ONLY; the production path draws internally with the
    contractual call order), applies log(U) computed in float32 then cast to
    float64, accepts iff log_u < log_alpha.  Returns ((x, u, g), accept, u_unif)."""
    if u_unif is None:
        u_unif = cp.random.random(x.shape[:1], dtype=cp.float32)
    accept = cp.log(u_unif).astype(cp.float64) < log_alpha
    accept_acc += accept.mean()
    if diag is not None:
        reject = ~accept
        diag[0] += reject.sum()
        diag[1] += cp.where(reject, (p0 * p0).mean(axis=-1), 0.0).sum()
        diag[2] += (p0 * p0).mean()
    accn = accept[:, None]
    return (cp.where(accn, xf, x), cp.where(accept, uf, u),
            cp.where(accn, gf, g)), accept, u_unif


def jitter_hmc_step(x, u, g, p0, eps, l_steps, potential, accept_acc, diag=None):
    """x,u,g: (W,dim),(W,),(W,dim); p0: refresh noise; l_steps: (W,) int.

    Recomposed propose + accept.  The totalized log_alpha produces bitwise-
    identical acceptance decisions and trajectories to the pre-refactor
    kernel for finite H0 - H and for -inf (log(U) < 0 almost surely; NaN
    comparisons rejected before); the H0 - H = +inf case (legacy accepted)
    now rejects -- registered as outside legacy equivalence and an S-7
    defect tripwire."""
    xf, uf, gf, pf, log_alpha, _alpha = jitter_propose(x, u, g, p0, eps, l_steps,
                                                       potential)
    (xn, un, gn), _accept, _uu = jitter_accept(x, u, g, xf, uf, gf, log_alpha,
                                               accept_acc, p0=p0, diag=diag)
    return xn, un, gn


# ---------------------------------------------------------------------------
# Multinomial NUTS, iterative, vectorized over walkers.
# ---------------------------------------------------------------------------

def popcount(k):
    return bin(k).count("1")


def trailing_ones(k):
    t = 0
    while k & 1:
        t += 1
        k >>= 1
    return t


def nuts_step(x, u, g, p0, eps, max_depth, potential, stats):
    w_n, dim = x.shape
    h0 = u.astype(cp.float64) + 0.5 * dot64(p0, p0)
    xm, pm = x.copy(), p0.copy()
    xp, pp = x.copy(), p0.copy()
    um, gm = u.copy(), g.copy()
    up, gp = u.copy(), g.copy()
    xprop, uprop, gprop = x.copy(), u.copy(), g.copy()
    log_s = cp.zeros(w_n, dtype=cp.float64)
    active = cp.ones(w_n, dtype=bool)
    divergent = cp.zeros(w_n, dtype=bool)
    depth_done = cp.zeros(w_n, dtype=cp.int32)
    ck_x = cp.zeros((max_depth, w_n, dim), dtype=cp.float32)
    ck_p = cp.zeros((max_depth, w_n, dim), dtype=cp.float32)
    grads_used = 0
    for j in range(max_depth):
        if not bool(active.any()):
            break
        d = cp.where(cp.random.random(w_n, dtype=cp.float32) < 0.5, 1.0, -1.0).astype(cp.float32)
        sel = (d > 0)
        bx = cp.where(sel[:, None], xp, xm)
        bp = cp.where(sel[:, None], pp, pm)
        bg = cp.where(sel[:, None], gp, gm)
        bu = cp.where(sel, up, um)
        ed = (eps * d)[:, None] if d.ndim else eps * d
        ed = eps * d[:, None]
        ph = bp - 0.5 * ed * bg
        st_logs = cp.full(w_n, -cp.inf, dtype=cp.float64)
        st_turn = cp.zeros(w_n, dtype=bool)
        st_div = cp.zeros(w_n, dtype=bool)
        sx, su, sg = bx.copy(), bu.copy(), bg.copy()
        pend = cp.zeros_like(p0)
        n_steps = 1 << j
        for k in range(n_steps):
            mask = active & ~st_turn & ~st_div
            if not bool(mask.any()):
                break
            grads_used += 1
            bx = cp.where(mask[:, None], bx + ed * ph, bx)
            ut, gt = potential(bx)
            bu = cp.where(mask, ut, bu)
            bg = cp.where(mask[:, None], gt, bg)
            pk = ph - 0.5 * ed * bg
            h = bu.astype(cp.float64) + 0.5 * dot64(pk, pk)
            dh = h0 - h
            div_now = ~cp.isfinite(dh) | (dh < -DIVERGENCE)
            st_div |= mask & div_now
            ok = mask & ~div_now
            # Stan-style: divergent leaves count in the denominator with
            # zero contribution.
            stats["accept_sum"] += float(cp.where(ok, cp.minimum(1.0, cp.exp(dh)), 0.0).sum().get())
            stats["accept_n"] += int(mask.sum().get())
            stats["walker_grads"] += int(mask.sum().get())
            new_logs = cp.logaddexp(st_logs, cp.where(ok, dh, -cp.inf))
            take = ok & (cp.log(cp.random.random(w_n, dtype=cp.float32)) < (dh - new_logs))
            sx = cp.where(take[:, None], bx, sx)
            su = cp.where(take, bu, su)
            sg = cp.where(take[:, None], bg, sg)
            st_logs = cp.where(ok, new_logs, st_logs)
            pend = cp.where(ok[:, None], pk, pend)
            if k % 2 == 0:
                slot = popcount(k)
                ck_x[slot] = cp.where(ok[:, None], bx, ck_x[slot])
                ck_p[slot] = cp.where(ok[:, None], pk, ck_p[slot])
            else:
                pc = popcount(k)
                for m in range(1, trailing_ones(k) + 1):
                    slot = pc - m
                    dx = d[:, None] * (bx - ck_x[slot])
                    turn_blk = (dot(dx, ck_p[slot]) < 0) | (dot(dx, pk) < 0)
                    st_turn |= ok & turn_blk
            ph = cp.where(ok[:, None], ph - ed * bg, ph)
        sub_ok = active & ~st_turn & ~st_div
        divergent |= active & st_div
        upd = sub_ok & (cp.log(cp.random.random(w_n, dtype=cp.float32)) < (st_logs - log_s))
        xprop = cp.where(upd[:, None], sx, xprop)
        uprop = cp.where(upd, su, uprop)
        gprop = cp.where(upd[:, None], sg, gprop)
        log_s = cp.where(sub_ok, cp.logaddexp(log_s, st_logs), log_s)
        fwd = sub_ok & sel
        bwd = sub_ok & ~sel
        xp = cp.where(fwd[:, None], bx, xp)
        pp = cp.where(fwd[:, None], pend, pp)
        up = cp.where(fwd, bu, up)
        gp = cp.where(fwd[:, None], bg, gp)
        xm = cp.where(bwd[:, None], bx, xm)
        pm = cp.where(bwd[:, None], pend, pm)
        um = cp.where(bwd, bu, um)
        gm = cp.where(bwd[:, None], bg, gm)
        dxo = xp - xm
        outer_turn = (dot(dxo, pm) < 0) | (dot(dxo, pp) < 0)
        depth_done = cp.where(sub_ok, j + 1, depth_done)
        active = sub_ok & ~outer_turn
    stats["depth_sum"] += float(depth_done.sum().get())
    stats["depth_n"] += int(depth_done.size)
    stats["saturated"] += int(active.sum().get())
    stats["divergent"] += int(divergent.sum().get())
    stats["grads"] += grads_used
    return xprop, uprop, gprop


# ---------------------------------------------------------------------------
# Selftest: exact Gaussian target.
# ---------------------------------------------------------------------------

def selftest():
    cp.random.seed(1240001)
    dim = 8
    lams = np.logspace(np.log10(0.25), np.log10(4.0), dim)
    potential = gaussian_potential(lams)
    w_n = 512
    x = cp.zeros((w_n, dim), dtype=cp.float32)
    u, g = potential(x)
    eps = 0.25
    steps, burnin = 3000, 500
    stats = new_stats()
    acc1 = cp.zeros(dim, dtype=cp.float64)
    acc2 = cp.zeros(dim, dtype=cp.float64)
    kept = 0
    for step in range(steps):
        p0 = cp.random.standard_normal(x.shape, dtype=cp.float32)
        x, u, g = nuts_step(x, u, g, p0, eps, MAX_DEPTH, potential, stats)
        if step >= burnin:
            acc1 += x.mean(axis=0)
            acc2 += (x * x).mean(axis=0)
            kept += 1
    mean = (acc1 / kept).get()
    second = (acc2 / kept).get()
    var_exact = 1.0 / lams
    # Effective sample size is unknown; bound the MC error using independent
    # walkers only (kept steps are correlated): se(mean) per coord <=
    # sqrt(var / w_n) (walker average at one step), improved by averaging over
    # steps -- use the conservative single-step bound.
    se_mean = np.sqrt(var_exact / w_n)
    z_mean = np.abs(mean) / se_mean
    # var of per-coord second moment across walkers ~ 2 var^2 / w_n
    se_second = np.sqrt(2.0 * var_exact ** 2 / w_n)
    z_second = np.abs(second - var_exact) / se_second
    mean_depth = stats["depth_sum"] / stats["depth_n"]
    accept_stat = stats["accept_sum"] / stats["accept_n"]
    print(f"selftest gaussian: max|z_mean|={z_mean.max():.2f} max|z_second|={z_second.max():.2f} "
          f"mean_depth={mean_depth:.2f} accept_stat={accept_stat:.3f} "
          f"divergent={stats['divergent']} saturated={stats['saturated']}")
    assert z_mean.max() < 4.0, f"NUTS mean off: z={z_mean}"
    assert z_second.max() < 4.0, f"NUTS second moment off: z={z_second}"
    assert stats["divergent"] == 0 and stats["saturated"] == 0
    assert mean_depth >= 2.0, f"depth too small to exercise checkpoint logic: {mean_depth}"

    # Tiny eps: energy nearly conserved -> accept_stat ~ 1 and depth grows.
    cp.random.seed(1240002)
    x = cp.zeros((64, dim), dtype=cp.float32)
    u, g = potential(x)
    stats2 = new_stats()
    for _ in range(30):
        p0 = cp.random.standard_normal(x.shape, dtype=cp.float32)
        x, u, g = nuts_step(x, u, g, p0, 0.01, MAX_DEPTH, potential, stats2)
    astat = stats2["accept_sum"] / stats2["accept_n"]
    print(f"selftest tiny-eps: accept_stat={astat:.5f} mean_depth={stats2['depth_sum']/stats2['depth_n']:.2f}")
    assert astat > 0.999, f"accept-stat should approach 1 at tiny eps: {astat}"

    # Marginal invariance: coupled refresh must not change per-chain law.
    def run_marg(method, seed):
        cp.random.seed(seed)
        reps, chains = 96, 8
        xs = cp.zeros((reps, chains, dim), dtype=cp.float32)
        us, gs = potential(xs.reshape(-1, dim))
        us = us.reshape(reps, chains)
        gs = gs.reshape(reps, chains, dim)
        st = new_stats()
        m2 = cp.zeros(dim, dtype=cp.float64)
        kept = 0
        for step in range(1500):
            p0 = make_noise(xs.shape, method).reshape(-1, dim)
            xf, uf, gf = nuts_step(xs.reshape(-1, dim), us.reshape(-1), gs.reshape(-1, dim),
                                   p0, eps, MAX_DEPTH, potential, st)
            xs = xf.reshape(reps, chains, dim)
            us = uf.reshape(reps, chains)
            gs = gf.reshape(reps, chains, dim)
            if step >= 300:
                m2 += (xs[:, 0, :] ** 2).mean(axis=0)
                kept += 1
        return (m2 / kept).get()

    m2_iid = run_marg("iid", 1240010)
    m2_zs = run_marg("zero_sum", 1240011)
    se = np.sqrt(2.0 * var_exact ** 2 / 96)
    z_inv = np.abs(m2_zs - m2_iid) / (se * np.sqrt(2.0))
    print(f"selftest marginal invariance: max|z|={z_inv.max():.2f}")
    assert z_inv.max() < 4.0, f"coupled NUTS changed the per-chain marginal: z={z_inv}"

    # --- jitter kernel exactness -------------------------------------------
    cp.random.seed(1240020)
    w_n = 8
    x0 = cp.random.standard_normal((w_n, dim), dtype=cp.float32) * 0.5
    p0 = cp.random.standard_normal((w_n, dim), dtype=cp.float32)
    _, g0 = potential(x0)
    eps_t = 0.2

    def plain_leapfrog(xs, gs, ps, l_val):
        ph = ps - 0.5 * eps_t * gs
        xt = xs.copy()
        for l in range(1, l_val + 1):
            xt = xt + eps_t * ph
            ut, gt = potential(xt)
            if l == l_val:
                pf = ph - 0.5 * eps_t * gt
            else:
                ph = ph - eps_t * gt
        return xt, ut, gt, pf

    l_const = cp.full(w_n, 7, dtype=cp.int32)
    va = jitter_leapfrog(x0, g0, p0, eps_t, l_const, potential)
    vb = plain_leapfrog(x0, g0, p0, 7)
    worst_const = max(float(cp.max(cp.abs(a - b)).get()) for a, b in zip(va, vb))
    print(f"selftest jitter constant-L vs plain leapfrog: max abs diff {worst_const:.3e}")
    assert worst_const < 1e-6, "constant-L jitter mismatch"

    l_mixed = cp.asarray([3, 3, 7, 7, 12, 12, 20, 20], dtype=cp.int32)
    vm = jitter_leapfrog(x0, g0, p0, eps_t, l_mixed, potential)
    worst_mixed = 0.0
    for l_val in (3, 7, 12, 20):
        sel = cp.asarray(l_mixed.get() == l_val)
        vs = jitter_leapfrog(x0[sel], g0[sel], p0[sel], eps_t,
                             cp.full(int(sel.sum()), l_val, dtype=cp.int32), potential)
        for a, b in zip((vm[0][sel], vm[1][sel], vm[2][sel], vm[3][sel]), vs):
            worst_mixed = max(worst_mixed, float(cp.max(cp.abs(a - b)).get()))
    print(f"selftest jitter mixed-L group equivalence: max abs diff {worst_mixed:.3e}")
    assert worst_mixed < 1e-6, "mixed-L masked continuation corrupts trajectories"

    # jittered MH kernel moment test on the Gaussian target
    cp.random.seed(1240030)
    rng_l = np.random.default_rng(1240031)
    xj = cp.zeros((512, dim), dtype=cp.float32)
    uj, gj = potential(xj)
    acc = cp.zeros((), dtype=cp.float32)
    m1 = cp.zeros(dim, dtype=cp.float64)
    m2 = cp.zeros(dim, dtype=cp.float64)
    kept = 0
    for step in range(1500):
        p0j = cp.random.standard_normal(xj.shape, dtype=cp.float32)
        lj = cp.asarray(rng_l.integers(1, 2 * LBAR + 1, size=512), dtype=cp.int32)
        xj, uj, gj = jitter_hmc_step(xj, uj, gj, p0j, eps, lj, potential, acc)
        if step >= 300:
            m1 += xj.mean(axis=0)
            m2 += (xj * xj).mean(axis=0)
            kept += 1
    mean_j = (m1 / kept).get()
    second_j = (m2 / kept).get()
    zj_mean = np.abs(mean_j) / np.sqrt(var_exact / 512)
    zj_sec = np.abs(second_j - var_exact) / np.sqrt(2.0 * var_exact ** 2 / 512)
    print(f"selftest jitter MH moments: max|z_mean|={zj_mean.max():.2f} "
          f"max|z_second|={zj_sec.max():.2f} accept={float((acc/1500).get()):.3f}")
    assert zj_mean.max() < 4.0 and zj_sec.max() < 4.0, "jitter MH kernel moments off"

    # --- NUTS selection distribution: 1-D quantiles ------------------------
    cp.random.seed(1240040)
    pot1 = gaussian_potential(np.array([1.0]))
    xq = cp.zeros((512, 1), dtype=cp.float32)
    uq, gq = pot1(xq)
    stq = new_stats()
    samples = []
    for step in range(3000):
        p0q = cp.random.standard_normal(xq.shape, dtype=cp.float32)
        xq, uq, gq = nuts_step(xq, uq, gq, p0q, 0.35, MAX_DEPTH, pot1, stq)
        if step >= 500 and step % 100 == 0:
            samples.append(xq[:, 0].get().copy())
    pool = np.concatenate(samples)
    from scipy.stats import norm as scipy_norm
    worst_q = 0.0
    n_eff = pool.size / 2.0  # conservative: halve for residual correlation
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        zq = scipy_norm.ppf(q)
        se = math.sqrt(q * (1 - q) / n_eff) / scipy_norm.pdf(zq)
        worst_q = max(worst_q, abs(float(np.quantile(pool, q)) - zq) / se)
    print(f"selftest NUTS 1-D quantiles: max|z|={worst_q:.2f} over 5 quantiles "
          f"(pool {pool.size})")
    assert worst_q < 4.0, "NUTS selection distribution off at the quantile level"
    print("selftest PASS")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def normalized_probe(dim, seed):
    rng = cp.random.RandomState(seed)
    probe = rng.standard_normal(dim, dtype=cp.float32)
    probe[0] = 0.0
    return probe / cp.linalg.norm(probe)


def draw_l_sequences(design):
    """Design-seeded L draws: shared across couplings within this design,
    independent across replicates (design_seed embeds the replicate).
    jitter_shared draws one L per (step, ensemble), broadcast only over
    that ensemble's K chains, so the across-ensemble variance metric
    marginalizes over schedules for BOTH jitter kernels."""
    rng = np.random.default_rng(design["design_seed"] + 777)
    total = design["ref_steps"] + design["steps"]
    shared = rng.integers(1, 2 * LBAR + 1, size=(total, design["reps"]))
    perchain = rng.integers(1, 2 * LBAR + 1, size=(total, design["reps"] * design["chains"]))
    ref_shared = rng.integers(1, 2 * LBAR + 1, size=(total, design["ref_reps"]))
    ref_perchain = rng.integers(1, 2 * LBAR + 1,
                                size=(total, design["ref_reps"] * design["ref_chains"]))
    return {"jitter_shared": (shared, ref_shared),
            "jitter_perchain": (perchain, ref_perchain)}


def kernel_step(kernel, x, u, g, p0, design, potential, step_idx, seqs, accept_acc,
                stats, diag=None, ref=False):
    w_n = x.shape[0]
    eps = design["eps"]
    if kernel in ("jitter_shared", "jitter_perchain"):
        seq = seqs[kernel][1 if ref else 0]
        chains = design["ref_chains"] if ref else design["chains"]
        row = seq[step_idx]
        if kernel == "jitter_shared":
            row = np.repeat(row, chains)
        l_steps = cp.asarray(row[:w_n], dtype=cp.int32)
        stats["l_sum"] += float(row[:w_n].sum())
        stats["l_n"] += w_n
        return jitter_hmc_step(x, u, g, p0, eps, l_steps, potential, accept_acc, diag)
    if kernel == "nuts":
        return nuts_step(x, u, g, p0, eps, MAX_DEPTH, potential, stats)
    raise ValueError(kernel)


def new_stats():
    return dict(accept_sum=0.0, accept_n=0, depth_sum=0.0, depth_n=0,
                saturated=0, divergent=0, grads=0, walker_grads=0,
                l_sum=0.0, l_n=0)


def reference_mean(kernel, design, potential, seqs):
    cp.random.seed(design["seed"] + 77717)
    w_n = design["ref_reps"] * design["ref_chains"]
    dim = design["dim"]
    x = cp.zeros((w_n, dim), dtype=cp.float32)
    u, g = potential(x)
    acc = cp.zeros(dim, dtype=cp.float32)
    accept_acc = cp.zeros((), dtype=cp.float32)
    stats = new_stats()
    windows = 0
    for step in range(design["ref_steps"]):
        p0 = cp.random.standard_normal(x.shape, dtype=cp.float32)
        x, u, g = kernel_step(kernel, x, u, g, p0, design, potential, step, seqs,
                              accept_acc, stats, ref=True)
        if step >= design["ref_burnin"]:
            acc += x.mean(axis=0)
            windows += 1
    return acc / windows


def run_method(kernel, design, potential, ref, method, seqs):
    cp.random.seed(design["seed"] + 101 * (METHODS.index(method) + 1) + 50011)
    reps, chains, dim = design["reps"], design["chains"], design["dim"]
    x = cp.zeros((reps, chains, dim), dtype=cp.float32)
    u, g = potential(x.reshape(-1, dim))
    ens_acc = cp.zeros((reps, dim), dtype=cp.float32)
    single_acc = cp.zeros((reps, dim), dtype=cp.float32)
    windows = 0
    accept_acc = cp.zeros((), dtype=cp.float32)
    stats = new_stats()
    diag = cp.zeros(3, dtype=cp.float64)
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for step in range(design["ref_steps"], design["ref_steps"] + design["steps"]):
        p0 = make_noise(x.shape, method).reshape(-1, dim)
        xf, uf, gf = kernel_step(kernel, x.reshape(-1, dim), u, g, p0, design, potential,
                                 step, seqs, accept_acc, stats, diag)
        x = xf.reshape(reps, chains, dim)
        u, g = uf, gf
        if step - design["ref_steps"] >= design["burnin"]:
            ens_acc += x.mean(axis=1)
            single_acc += x[:, 0, :]
            windows += 1
    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    ens = ens_acc / windows
    single = single_acc / windows
    ens_err = ens - ref[None, :]
    single_err = single - ref[None, :]
    ens_centered = ens - ens.mean(axis=0)[None, :]
    single_centered = single - single.mean(axis=0)[None, :]
    row = {
        **design,
        "target": "real_logistic_nuts",
        "kernel": kernel,
        "method": method,
        "coupling": method,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "git_hash": git_hash(),
        "elapsed_sec": elapsed,
        "n_ensembles": reps,
        "windows": windows,
        "mse": float(cp.mean(cp.sum(ens_err * ens_err, axis=1)).get()),
        "trace_var": float(cp.mean(cp.sum(ens_centered * ens_centered, axis=1)).get()),
        "single_mse": float(cp.mean(cp.sum(single_err * single_err, axis=1)).get()),
        "single_trace_var": float(cp.mean(cp.sum(single_centered * single_centered, axis=1)).get()),
        "bias_sq": float(cp.sum((ens.mean(axis=0) - ref) ** 2).get()),
    }
    row["provenance"] = provenance()
    if kernel == "nuts":
        row.update({
            "accept_stat": stats["accept_sum"] / max(stats["accept_n"], 1),
            "mean_depth": stats["depth_sum"] / max(stats["depth_n"], 1),
            "divergent": stats["divergent"],
            "saturated_frac": stats["saturated"] / (design["steps"] * reps * chains),
            # lockstep batched potential calls vs realized per-walker
            # leapfrogs; the two differ under masking and must not be
            # conflated with equal-compute claims.
            "batched_calls_per_outer": stats["grads"] / design["steps"],
            "leapfrogs_per_walker_outer": stats["walker_grads"]
                                          / (design["steps"] * reps * chains),
        })
    else:
        row.update({
            "accept_rate": float((accept_acc / design["steps"]).get()),
            "reject_count": float(diag[0].get()),
            "mom_sq_given_reject": float((diag[1] / cp.maximum(diag[0], 1.0)).get()),
            "mom_sq_overall": float((diag[2] / design["steps"]).get()),
            "mean_l": stats["l_sum"] / max(stats["l_n"], 1),
        })
    return row


def calibrate_eps(dataset, dim, prior, data_x, data_y, chunk_rows,
                  pilot_steps=60, pilot_reps=8, pilot_chains=8):
    # Identical seeds/logic to hmc_real_logistic.calibrate_eps (L=10), so the
    # chosen eps matches the HMC campaign cell for cell.
    potential = logistic_potential(data_x, data_y, prior, chunk_rows)
    seed = {"wdbc": 710000, "spambase": 720000, "adult": 730000}[dataset] + int(round(prior * 1000)) + 424243
    chosen, chosen_rate, rates = None, None, {}
    for gi, eps in enumerate(EPS_SCAN):
        cp.random.seed(seed + 1000 * gi)
        x = cp.zeros((pilot_reps * pilot_chains, dim), dtype=cp.float32)
        u, g = potential(x)
        accept_acc = cp.zeros((), dtype=cp.float32)
        l_steps = cp.full(x.shape[0], LBAR, dtype=cp.int32)
        for _ in range(pilot_steps):
            p0 = cp.random.standard_normal(x.shape, dtype=cp.float32)
            x, u, g = jitter_hmc_step(x, u, g, p0, eps, l_steps, potential, accept_acc)
        rate = float((accept_acc / pilot_steps).get())
        rates[eps] = rate
        if chosen is None and 0.6 <= rate <= 0.9:
            chosen, chosen_rate = eps, rate
    if chosen is None:
        eps = min(rates, key=lambda e: abs(rates[e] - 0.75))
        chosen, chosen_rate = eps, rates[eps]
    print(json.dumps({"calibrate": dataset, "prior": prior,
                      "rates": {f"{k:.6g}": v for k, v in sorted(rates.items())},
                      "eps": chosen, "pilot_accept": chosen_rate}, sort_keys=True), flush=True)
    return chosen, chosen_rate


def make_designs(dataset, dim, args, eps_by_prior, pilot_by_prior):
    defaults = DATASET_DEFAULTS[dataset]
    chains = args.chains or [8]
    priors = args.priors or [0.1, 0.3]
    # Tuple-based block 31x,xxx,xxx: disjoint across dataset/replicate/design
    # and from every prior campaign (repo-wide max seed 20,260,712).
    ds_idx = {"wdbc": 0, "spambase": 1, "adult": 2}[dataset]
    seed_base = 310_000_000 + ds_idx * 10_000_000 + args.replicate * 100_000
    idx = 0
    for prior in priors:
        for k in chains:
            reps = args.reps or max(24, defaults["reps"] // max(1, k // 8))
            yield {
                "dataset": dataset,
                "dim": dim,
                "chains": k,
                "prior": prior,
                "eps": eps_by_prior[prior],
                "lbar": LBAR,
                "max_depth": MAX_DEPTH,
                "eps_calibrated": args.eps is None,
                "pilot_accept": pilot_by_prior[prior],
                "steps": args.steps or defaults["steps"],
                "burnin": args.burnin or defaults["burnin"],
                "ref_steps": args.ref_steps or defaults["ref_steps"],
                "ref_burnin": args.ref_burnin or defaults["ref_burnin"],
                "ref_reps": args.ref_reps,
                "ref_chains": args.ref_chains,
                "reps": reps,
                "replicate": args.replicate,
                "eps_condition": args.eps_condition,
                "quick": bool(args.quick),
                "seed": seed_base + idx,
                "design_seed": seed_base + 50_000 + idx,
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


def qc_check(rows):
    """Frozen-manifest QC: consistent designs, 3 replicates, full runs only."""
    for r in rows:
        assert not r.get("quick"), f"quick row in table inputs: {r.get('dataset')}"
    for (ds, prior, kernel, cond) in {(r["dataset"], r["prior"], r["kernel"],
                                       r.get("eps_condition", "primary")) for r in rows}:
        sub = [r for r in rows if (r["dataset"], r["prior"], r["kernel"],
                                   r.get("eps_condition", "primary")) == (ds, prior, kernel, cond)]
        reps_seen = sorted({r["replicate"] for r in sub})
        assert reps_seen == [0, 1, 2], f"{ds}/{prior}/{kernel}/{cond}: replicates {reps_seen}"
        for field in ("chains", "steps", "burnin", "reps", "eps"):
            vals = {r[field] for r in sub}
            assert len(vals) == 1, f"{ds}/{prior}/{kernel}/{cond}: {field} inconsistent {vals}"


def notch_required(rows, ds, prior):
    """Pre-registered trigger, from diagnostics alone (never from ratios):
    any zero_sum NUTS replicate at primary eps with saturated_frac > 0.20,
    divergent > 0.5% of steps*reps*chains, or a non-finite estimate."""
    sub = [r for r in rows if r["dataset"] == ds and r["prior"] == prior
           and r["kernel"] == "nuts" and r["method"] == "zero_sum"
           and r.get("eps_condition", "primary") == "primary"]
    for r in sub:
        walker_steps = r["steps"] * r["reps"] * r["chains"]
        if (r["saturated_frac"] > 0.20 or r["divergent"] > 0.005 * walker_steps
                or not np.isfinite(r["trace_var"])):
            return True
    return False


def make_table(table_inputs, table_out):
    rows = []
    for path in table_inputs:
        with Path(path).open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    qc_check(rows)
    conditions = sorted({r.get("eps_condition", "primary") for r in rows},
                        key=lambda c: (c != "primary", c))
    show_cond = len(conditions) > 1
    header = "Dataset & prior & kernel & "
    header += ("$\\epsilon$ cond.\\ & " if show_cond else "")
    header += "$C_\\star$/iid & pairwise/iid \\\\"
    lines = [
        f"\\begin{{tabular}}{{lll{'l' if show_cond else ''}rr}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    summary = ["== nuts_real.tex (trace-var ratios, median across 3 reps) =="]
    for ds in DATASETS:
        for prior in [0.1, 0.3]:
            if notch_required(rows, ds, prior):
                has_notch = any(r.get("eps_condition") == "notch" for r in rows
                                if r["dataset"] == ds and r["prior"] == prior)
                assert has_notch, (
                    f"NOTCH REQUIRED for {ds}/{prior} (pre-registered trigger fired) "
                    f"but no notch rows present: rerun all kernels/couplings at "
                    f"eps/sqrt2 with --eps-condition notch")
            for kernel in KERNELS:
                for cond in conditions:
                    key = {"dataset": ds, "prior": prior, "kernel": kernel,
                           "eps_condition": cond}
                    sub_exists = any(all(r.get(k) == v for k, v in key.items())
                                     for r in rows)
                    if not sub_exists:
                        continue
                    zs = ratios(rows, key, "zero_sum")
                    pa = ratios(rows, key, "antithetic")
                    cond_col = f"{cond} & " if show_cond else ""
                    lines.append(
                        f"{DATASET_LABELS[ds]} & {prior} & {KERNEL_LABELS[kernel]} & "
                        f"{cond_col}{cell(zs)} & {cell(pa)} \\\\"
                    )
                    extra = ""
                    if kernel == "nuts":
                        nr = [r for r in rows if all(r.get(k) == v for k, v in key.items())
                              and r["method"] == "zero_sum"]
                        extra = (f" depth={np.median([r['mean_depth'] for r in nr]):.2f}"
                                 f" accept={np.median([r['accept_stat'] for r in nr]):.2f}"
                                 f" sat={np.median([r['saturated_frac'] for r in nr]):.3f}"
                                 f" div={int(np.median([r['divergent'] for r in nr]))}")
                    summary.append(f"  {ds} p={prior} {kernel}/{cond}: "
                                   f"C*/iid={np.median(zs):.3g} "
                                   f"pair/iid={np.median(pa):.3g}{extra}")
            lines.append("\\addlinespace" if (ds, prior) != ("adult", 0.3) else "% end")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    Path(table_out).write_text("\n".join(lines), encoding="utf-8")
    # Pre-registered aggregate contrasts (H1 paired, H2 band), printed for the
    # result review; the paper text follows these numbers.
    deltas, nuts_meds = [], []
    for ds in DATASETS:
        for prior in [0.1, 0.3]:
            base = {"dataset": ds, "prior": prior, "eps_condition": "primary"}
            shared = np.median(ratios(rows, {**base, "kernel": "jitter_shared"}, "zero_sum"))
            perch = np.median(ratios(rows, {**base, "kernel": "jitter_perchain"}, "zero_sum"))
            nuts_meds.append(np.median(ratios(rows, {**base, "kernel": "nuts"}, "zero_sum")))
            deltas.append(perch - shared)
    h1 = sum(d > 0 for d in deltas) >= 5 and float(np.median(deltas)) >= 0.15
    h2 = float(np.median(nuts_meds)) > 0.8
    summary.append(f"  H1 paired contrast: positive in {sum(d > 0 for d in deltas)}/6, "
                   f"median delta {np.median(deltas):.3f} -> H1 {'HOLDS' if h1 else 'not confirmed'}")
    summary.append(f"  H2 NUTS six-cell median ratio {np.median(nuts_meds):.3f} -> "
                   f"H2 {'HOLDS (parity/hurts)' if h2 else 'REFUTED (coupling survives)'}")
    summary.append(f"  -> wrote {table_out}")
    print("\n".join(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--eps-condition", choices=["primary", "notch"], default="primary",
                        help="notch reruns the cell at eps/sqrt(2) (pre-registered trigger)")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--kernels", nargs="+", default=KERNELS)
    parser.add_argument("--chains", nargs="+", type=int, default=None)
    parser.add_argument("--priors", nargs="+", type=float, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--burnin", type=int, default=None)
    parser.add_argument("--ref-steps", type=int, default=None)
    parser.add_argument("--ref-burnin", type=int, default=None)
    parser.add_argument("--ref-reps", type=int, default=16)
    parser.add_argument("--ref-chains", type=int, default=16)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--make-table", action="store_true")
    parser.add_argument("--table-inputs", nargs="+", type=Path,
                        default=[Path(f"outputs/sweeps/nuts_real_rep{r}.jsonl") for r in range(3)])
    parser.add_argument("--table-out", type=Path, default=Path("paper/tables/nuts_real.tex"))
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
            data_x, data_y = load_dataset(args.data_dir, dataset)
            dim = int(data_x.shape[1])
            potential = logistic_potential(data_x, data_y, None, args.chunk_rows)
            priors = args.priors or [0.1, 0.3]
            eps_by_prior, pilot_by_prior = {}, {}
            for prior in priors:
                if args.eps is not None:
                    eps_by_prior[prior], pilot_by_prior[prior] = args.eps, None
                else:
                    eps_by_prior[prior], pilot_by_prior[prior] = calibrate_eps(
                        dataset, dim, prior, data_x, data_y, args.chunk_rows)
                if args.eps_condition == "notch":
                    eps_by_prior[prior] = eps_by_prior[prior] / math.sqrt(2.0)
            for design in make_designs(dataset, dim, args, eps_by_prior, pilot_by_prior):
                potential = logistic_potential(data_x, data_y, design["prior"], args.chunk_rows)
                seqs = draw_l_sequences(design)
                for kernel in args.kernels:
                    ref = reference_mean(kernel, design, potential, seqs)
                    for method in args.methods:
                        result = run_method(kernel, design, potential, ref, method, seqs)
                        f.write(json.dumps(result, sort_keys=True) + "\n")
                        f.flush()
                        print(json.dumps({k: v for k, v in result.items()
                                          if k not in ("gold_mean",)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
