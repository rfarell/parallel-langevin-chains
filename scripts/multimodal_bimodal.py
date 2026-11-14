"""Bimodal Gaussian-mixture multimodal study: coupled parallel ULA.

Modes:
  --selftest   gradient/coupling/determinism checks (abort campaign on fail)
  --pilot      EXPLORATORY calibration only (stability + transition rates);
               computes no scored metric.
  --campaign   the pre-registered campaign of
               campaigns/multimodal_bimodal_precommit.md. Refuses to run
               without the precommit file present (its sha256 goes into every
               row). --quick runs a tiny smoke (rows marked quick-smoke,
               refused by the emitter).
  --emit       validator + gates + pooled ratios + band words from the
               campaign JSONL. The emitter aborts on any validator or gate
               inconsistency; band words are printed by code, never by hand.

Target: pi(x) = w N(+a e1, I_d) + (1-w) N(-a e1, I_d), d = 10.
With r(x) = sigmoid(2 a x1 + log(w/(1-w))):  grad U(x) = x - (2 r(x) - 1) a e1.

Couplings clone scripts/nonlogconcave_real.py make_noise semantics
(iid / zero_sum / antithetic / common), ported cupy -> torch.
Seed blocks 95x,xxx,xxx (see precommit; disjoint from all prior campaigns).
"""

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch

METHODS = ["iid", "zero_sum", "antithetic", "common"]
SEED_BASE = 950_000_000          # pilot block
START_SEED_BASE = 951_000_000    # campaign starts (no method term)
NOISE_SEED_BASE = 952_000_000    # campaign noise (+ method_index)

PILOT_SEPARATIONS = [1.25, 1.5, 2.0, 2.5, 3.0]
PILOT_ETAS = [0.1, 0.25, 0.5]
PILOT_WEIGHTS = [0.5]
PILOT_ASYM = (2.0, 0.7)

# Registered cells, fixed order (precommit "Target" section).
CELLS = [
    {"name": "mild",         "a": 2.0, "w": 0.5, "K": 8,  "eta": 0.25, "cold": False},
    {"name": "intermediate", "a": 3.5, "w": 0.5, "K": 8,  "eta": 0.25, "cold": False},
    {"name": "metastable",   "a": 4.0, "w": 0.5, "K": 8,  "eta": 0.25, "cold": False},
    {"name": "asym",         "a": 2.0, "w": 0.7, "K": 8,  "eta": 0.25, "cold": False},
    {"name": "coldstart",    "a": 4.0, "w": 0.5, "K": 8,  "eta": 0.25, "cold": True},
    {"name": "k16",          "a": 4.0, "w": 0.5, "K": 16, "eta": 0.25, "cold": False},
    {"name": "eta01",        "a": 2.0, "w": 0.5, "K": 8,  "eta": 0.1,  "cold": False},
]
T_STEPS = 200_000
G_ENS = 256
DIM = 10
CHECKPOINTS = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
REPS = [0, 1, 2]
NORM_CEILING = 1e4
BANDS = [(0.5, "transfers"), (0.8, "partial"), (1.25, "parity"), (float("inf"), "hurts")]


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_facts():
    facts = {"gpu_name": "cpu", "gpu_uuid": "none", "driver_version": "none"}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version",
             "--format=csv,noheader"], text=True).strip().split("\n")[0]
        name, uuid, drv = [s.strip() for s in out.split(",")]
        facts = {"gpu_name": name, "gpu_uuid": uuid, "driver_version": drv}
    except Exception:
        if torch.cuda.is_available():
            facts["gpu_name"] = torch.cuda.get_device_name(0)
    return facts


def provenance(precommit_path=None):
    p = {
        "argv": " ".join(sys.argv),
        "git_hash": git_hash(),
        "script_sha256": sha256_of(__file__),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda or "none",
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "python": sys.version.split()[0],
        **gpu_facts(),
    }
    if precommit_path:
        p["precommit_sha256"] = sha256_of(precommit_path)
    return p


def make_noise(shape, method, generator, device, dtype=torch.float32):
    reps, chains, dim = shape
    def randn(*s):
        return torch.randn(*s, generator=generator, device=device, dtype=dtype)
    if method == "iid":
        return randn(reps, chains, dim)
    if method == "zero_sum":
        z = randn(reps, chains, dim)
        z = z - z.mean(dim=1, keepdim=True)
        return z * math.sqrt(chains / (chains - 1))
    if method == "common":
        z = randn(reps, 1, dim)
        return z.expand(reps, chains, dim).clone()
    if method == "antithetic":
        if chains % 2:
            z = randn(reps, chains // 2, dim)
            extra = randn(reps, 1, dim)
            return torch.cat([z, -z, extra], dim=1)
        z = randn(reps, chains // 2, dim)
        return torch.cat([z, -z], dim=1)
    raise ValueError(f"unknown method {method}")


def grad_u(x, a, logit_w):
    r = torch.sigmoid(2.0 * a * x[..., 0] + logit_w)
    g = x.clone()
    g[..., 0] = g[..., 0] - (2.0 * r - 1.0) * a
    return g


def rnd(t, sig=8):
    return [float(f"{v:.{sig}g}") for v in t.flatten().tolist()]


# ---------------------------------------------------------------- pilot mode

def run_cell(a, w, eta, steps, burnin, ensembles, chains, dim, method,
             seed, device):
    """Pilot: stability + transition stats only (hysteresis at +/- a/2)."""
    dtype = torch.float32
    logit_w = math.log(w / (1.0 - w))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = make_noise((ensembles, chains, dim), "iid", gen, device, dtype)
    signs = torch.where(
        torch.rand(ensembles, chains, generator=gen, device=device) < w,
        1.0, -1.0)
    x[..., 0] = x[..., 0] + signs * a
    label = torch.sign(x[..., 0])
    label[label == 0] = 1.0
    transitions = torch.zeros(ensembles, chains, device=device)
    sq_max, nonfinite = 0.0, 0
    root = math.sqrt(2.0 * eta)
    t0 = time.time()
    for t in range(steps):
        xi = make_noise((ensembles, chains, dim), method, gen, device, dtype)
        x = x - eta * grad_u(x, a, logit_w) + root * xi
        x1 = x[..., 0]
        flip_up = (label < 0) & (x1 > a / 2)
        flip_dn = (label > 0) & (x1 < -a / 2)
        if t >= burnin:
            transitions += flip_up.float() + flip_dn.float()
        label = torch.where(flip_up, torch.ones_like(label), label)
        label = torch.where(flip_dn, -torch.ones_like(label), label)
        if (t + 1) % max(1, steps // 4) == 0:
            sq_max = max(sq_max, float((x * x).sum(dim=-1).max()))
            nonfinite += int((~torch.isfinite(x)).sum())
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tr = transitions.flatten()
    q = torch.quantile(tr, torch.tensor([0.25, 0.5, 0.75], device=device))
    return {
        "transitions_mean": round(float(tr.mean()), 4),
        "transitions_q25": float(q[0]), "transitions_q50": float(q[1]),
        "transitions_q75": float(q[2]),
        "frac_chains_ge1": round(float((tr >= 1).float().mean()), 4),
        "max_sq_norm": round(sq_max, 2), "nonfinite": nonfinite,
        "steps_per_sec": round(steps / dt, 1), "wall_sec": round(dt, 2),
    }


# ------------------------------------------------------------- campaign mode

def campaign_cell(ci, method, rep, device, dtype, T, G, quick=False):
    """One accepted (cell, coupling, replicate) run. Returns the JSONL row.

    All accumulators are float64 on device (precommit Numerics); the only
    host syncs are the state hash at t=100, the cold-start checkpoints, and
    the end of run.
    """
    cell = CELLS[ci]
    a, w, K, eta, cold = (cell["a"], cell["w"], cell["K"], cell["eta"],
                          cell["cold"])
    logit_w = math.log(w / (1.0 - w))
    mi = METHODS.index(method)
    start_seed = START_SEED_BASE + 100_000 * rep + 1_000 * ci
    noise_seed = NOISE_SEED_BASE + 100_000 * rep + 1_000 * ci + mi
    gen_start = torch.Generator(device=device)
    gen_start.manual_seed(start_seed)
    gen_noise = torch.Generator(device=device)
    gen_noise.manual_seed(noise_seed)

    # starts: CRN across methods (gen_start carries no method term)
    z = torch.randn(G, K, DIM, generator=gen_start, device=device,
                    dtype=dtype)
    if cold:
        x = z.clone()
        x[..., 0] = x[..., 0] - a
        burn = 0
    else:
        signs = torch.where(
            torch.rand(G, K, generator=gen_start, device=device) < w,
            1.0, -1.0)
        x = z.clone()
        x[..., 0] = x[..., 0] + signs * a
        burn = T // 2
    kept = T - burn
    half = burn + kept // 2

    label = torch.sign(x[..., 0])
    label[label == 0] = 1.0
    f64 = torch.float64
    occ_sum = torch.zeros(G, device=device, dtype=f64)       # 1{x1>0} kept
    occ_all_sum = torch.zeros(G, device=device, dtype=f64)   # from t=0 (cold)
    hyst_sum = torch.zeros(G, device=device, dtype=f64)
    f2_sum = torch.zeros((), device=device, dtype=f64)       # pooled f^2
    f1_sum = torch.zeros((), device=device, dtype=f64)       # pooled f
    mean_sum = torch.zeros(G, DIM, device=device, dtype=f64)
    sec_x1 = torch.zeros(G, K, device=device, dtype=f64)
    sec_all = torch.zeros(G, K, device=device, dtype=f64)
    half1 = torch.zeros(G, device=device, dtype=f64)
    half2 = torch.zeros(G, device=device, dtype=f64)
    transitions = torch.zeros(G, K, device=device, dtype=f64)
    trans_all = torch.zeros(G, K, device=device, dtype=f64)
    fpt = torch.full((G, K), -1, device=device, dtype=torch.int64)
    ever_up = torch.zeros(G, K, device=device, dtype=torch.bool)
    ckpt_data = {}
    state_hash_100 = None
    sq_max, nonfinite = 0.0, 0
    root = math.sqrt(2.0 * eta)
    checkpoints = [c for c in CHECKPOINTS if c <= T]

    t0 = time.time()
    for t in range(T):
        xi = make_noise((G, K, DIM), method, gen_noise, device, dtype)
        x = x - eta * grad_u(x, a, logit_w) + root * xi
        x1 = x[..., 0]
        right = x1 > 0
        occ_t = right.to(f64).mean(dim=1)                    # (G,)
        flip_up = (label < 0) & (x1 > a / 2)
        flip_dn = (label > 0) & (x1 < -a / 2)
        flips = flip_up.to(f64) + flip_dn.to(f64)
        trans_all += flips
        newly = (fpt < 0) & (flip_up | flip_dn)
        fpt = torch.where(newly, torch.full_like(fpt, t), fpt)
        ever_up |= flip_up
        label = torch.where(flip_up, torch.ones_like(label), label)
        label = torch.where(flip_dn, -torch.ones_like(label), label)
        occ_all_sum += occ_t
        if t >= burn:
            transitions += flips
            occ_sum += occ_t
            hyst_sum += (label.to(f64) + 1.0).mean(dim=1) / 2.0
            f1_sum += occ_t.sum()
            f2_sum += (occ_t * occ_t).sum()
            mean_sum += x.mean(dim=1).to(f64)
            sec_x1 += (x1 * x1).to(f64)
            sec_all += (x * x).sum(dim=-1).to(f64)
            if t < half:
                half1 += occ_t
            else:
                half2 += occ_t
        if t == 99:
            state_hash_100 = hashlib.sha256(
                x.detach().to("cpu", torch.float64).numpy().tobytes()
            ).hexdigest()[:16]
        if (t + 1) in checkpoints:
            sq_max = max(sq_max, float((x * x).sum(dim=-1).max()))
            nonfinite += int((~torch.isfinite(x)).sum())
            if cold:
                any_right = right.any(dim=1)
                any_escape = ever_up.any(dim=1)
                ckpt_data[str(t + 1)] = {
                    "fbar_running": rnd(occ_all_sum / (t + 1)),
                    "f_inst": rnd(occ_t),
                    "any_right_frac": round(float(any_right.to(f64).mean()), 6),
                    "zero_escape_frac": round(
                        1.0 - float(any_escape.to(f64).mean()), 6),
                }
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    n_pool = kept * G
    row = {
        "status": "quick-smoke" if quick else (
            "float64-robustness" if dtype == torch.float64 else "accepted"),
        "cell_index": ci, "cell_name": cell["name"],
        "a": a, "w": w, "K": K, "eta": eta, "T": T, "burnin": burn,
        "G": G, "dim": DIM, "method": method, "replicate": rep,
        "start_seed": start_seed, "noise_seed": noise_seed,
        "dtype_dynamics": str(dtype).split(".")[-1],
        "dtype_accum": "float64",
        "state_hash_100": state_hash_100,
        "max_sq_norm": round(sq_max, 2), "nonfinite": nonfinite,
        "wall_sec": round(dt, 2), "steps_per_sec": round(T / dt, 1),
        "f_bar": rnd(occ_sum / kept),
        "f_bar_hyst": rnd(hyst_sum / kept),
        "f_pool_sum": float(f1_sum), "f2_pool_sum": float(f2_sum),
        "f_pool_n": n_pool,
        "mean_vec": rnd(mean_sum / kept),
        "sec_x1": rnd(sec_x1 / kept, 7),
        "sec_all_per_dim": rnd(sec_all / (kept * DIM), 7),
        "half1_occ": rnd(half1 / max(1, kept // 2)),
        "half2_occ": rnd(half2 / max(1, kept - kept // 2)),
        "transitions_kept": [int(v) for v in transitions.flatten().tolist()],
        "transitions_all": [int(v) for v in trans_all.flatten().tolist()],
        "fpt": [int(v) for v in fpt.flatten().tolist()],
    }
    if cold:
        row["checkpoints"] = ckpt_data
    return row


# ------------------------------------------------------------------ selftest

def selftest(device):
    ok = True
    e1 = torch.zeros(DIM, dtype=torch.float64)
    e1[0] = 1.0

    def u(y, a, w):
        comp = torch.stack([
            -0.5 * ((y - a * e1) ** 2).sum() + math.log(w),
            -0.5 * ((y + a * e1) ** 2).sum() + math.log(1 - w)])
        return -torch.logsumexp(comp, dim=0)

    torch.manual_seed(1)
    for w in (0.5, 0.7):
        a = 4.0
        logit_w = math.log(w / (1.0 - w))
        pts = [torch.randn(DIM, dtype=torch.float64) * 2.0 for _ in range(14)]
        for s in (0.0, 0.05, -0.05):                      # saddle region
            p = torch.randn(DIM, dtype=torch.float64)
            p[0] = s
            pts.append(p)
        for s in (3 * a, -3 * a):                         # saturated
            p = torch.randn(DIM, dtype=torch.float64)
            p[0] = s
            pts.append(p)
        assert len(pts) >= 19
        for p in pts:
            g = grad_u(p.view(1, 1, -1), a, logit_w)[0, 0]
            eps = 1e-6
            for j in (0, 1, DIM - 1):
                pp, pm = p.clone(), p.clone()
                pp[j] += eps
                pm[j] -= eps
                fd = float((u(pp, a, w) - u(pm, a, w)) / (2 * eps))
                rel = abs(fd - float(g[j])) / max(1.0, abs(fd))
                if rel > 1e-5:
                    print(f"SELFTEST FAIL grad w={w} j={j} rel={rel:.2e}")
                    ok = False

    gen = torch.Generator(device=device)
    gen.manual_seed(2)
    z = make_noise((40000, 8, 1), "zero_sum", gen, device)
    if float(z.sum(dim=1).abs().max()) > 1e-3:
        print("SELFTEST FAIL zero_sum row sums")
        ok = False
    if abs(float(z.var()) - 1.0) > 0.02:
        print(f"SELFTEST FAIL zero_sum marginal var {float(z.var()):.4f}")
        ok = False
    z0, z1 = z[:, 0, 0], z[:, 1, 0]
    corr = float(((z0 - z0.mean()) * (z1 - z1.mean())).mean()
                 / (z0.std() * z1.std()))
    if abs(corr - (-1.0 / 7.0)) > 0.02:
        print(f"SELFTEST FAIL C_star cross-corr {corr:.4f} vs {-1/7:.4f}")
        ok = False
    for K in (8, 16, 7):
        za = make_noise((512, K, 4), "antithetic", gen, device)
        k2 = K // 2
        if float((za[:, :k2] + za[:, k2:2 * k2]).abs().max()) > 1e-6:
            print(f"SELFTEST FAIL antithetic pairing K={K}")
            ok = False
        if K % 2 == 0 and float(za.sum(dim=1).abs().max()) > 1e-5:
            print(f"SELFTEST FAIL antithetic zero-sum at even K={K}")
            ok = False
    zc = make_noise((64, 8, 4), "common", gen, device)
    if float((zc - zc[:, :1]).abs().max()) > 1e-6:
        print("SELFTEST FAIL common broadcast")
        ok = False
    if abs(float(zc[:, 0].var()) - 1.0) > 0.1:
        print("SELFTEST FAIL common marginal var")
        ok = False

    r1 = campaign_cell(0, "zero_sum", 0, device, torch.float32,
                       T=200, G=8, quick=True)
    r2 = campaign_cell(0, "zero_sum", 0, device, torch.float32,
                       T=200, G=8, quick=True)
    if r1["state_hash_100"] != r2["state_hash_100"] or \
            r1["f_bar"] != r2["f_bar"]:
        print("SELFTEST FAIL determinism (same seeds differ)")
        ok = False

    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------- emitter

def pooled_ratio(vals_m, vals_iid, reps_m, reps_iid):
    """Pooled variance ratio + delta-method CI on log ratio.

    vals: list of (replicate, [per-ensemble f_bar]) pairs. Centering is per
    replicate; q = centered^2; ratio of mean(q); SE(log R) from cv^2/n.
    """
    def q_of(pairs):
        qs = []
        for _, v in pairs:
            t = torch.tensor(v, dtype=torch.float64)
            qs.append((t - t.mean()) ** 2)
        return torch.cat(qs)
    qm, qi = q_of(vals_m), q_of(vals_iid)
    n = len(qm)
    dof_scale = n / (n - len(vals_m))          # (G-1) per replicate
    s2m, s2i = float(qm.mean() * dof_scale), float(qi.mean() * dof_scale)
    if s2i == 0:
        return None
    ratio = s2m / s2i
    cv2m = float(qm.var() / qm.mean() ** 2) if qm.mean() > 0 else 0.0
    cv2i = float(qi.var() / qi.mean() ** 2) if qi.mean() > 0 else 0.0
    se_log = math.sqrt(cv2m / n + cv2i / n)
    lo, hi = ratio * math.exp(-2 * se_log), ratio * math.exp(2 * se_log)
    per_rep = []
    for (_, vm), (_, vi) in zip(vals_m, vals_iid):
        tm = torch.tensor(vm, dtype=torch.float64)
        ti = torch.tensor(vi, dtype=torch.float64)
        if float(ti.var()) > 0:
            per_rep.append(float(tm.var() / ti.var()))
    return {"ratio": ratio, "lo": lo, "hi": hi,
            "g2_m": cv2m - 2, "g2_iid": cv2i - 2,
            "per_rep": per_rep, "n": n}


def band_of(x):
    for ub, name in BANDS:
        if x <= ub if name != "parity" else x < ub:
            return name
    return "hurts"


def band_word(pr):
    if pr is None:
        return "degenerate"
    b_point = band_of(pr["ratio"])
    if band_of(pr["lo"]) != b_point or band_of(pr["hi"]) != b_point:
        return "unresolved"
    bands_rep = {band_of(r) for r in pr["per_rep"]}
    if len(bands_rep) > 1:
        return "unresolved"
    return b_point


def four_sigma(vals_m, vals_iid):
    """Across-ensemble 4-sigma comparison of per-ensemble statistics."""
    tm = torch.tensor([v for _, vs in vals_m for v in vs], dtype=torch.float64)
    ti = torch.tensor([v for _, vs in vals_iid for v in vs],
                      dtype=torch.float64)
    se = math.sqrt(float(tm.var()) / len(tm) + float(ti.var()) / len(ti))
    diff = abs(float(tm.mean()) - float(ti.mean()))
    return diff, se, (diff <= 4 * se if se > 0 else diff == 0)


def emit(args):
    rows = [json.loads(l) for l in open(args.out)]
    acc = [r for r in rows if r["status"] == "accepted"]
    f64r = [r for r in rows if r["status"] == "float64-robustness"]
    pre_sha = sha256_of(args.precommit)
    fail = []

    # ---- canonical validator
    want = {(ci, m, rep) for ci in range(len(CELLS)) for m in METHODS
            for rep in REPS}
    got = {(r["cell_index"], r["method"], r["replicate"]) for r in acc}
    if got != want:
        fail.append(f"validator: row inventory mismatch "
                    f"(missing {sorted(want - got)[:4]}, "
                    f"extra {sorted(got - want)[:4]})")
    if {(r["cell_index"], r["method"]) for r in f64r} != \
            {(2, m) for m in METHODS} or any(r["replicate"] != 0 for r in f64r):
        fail.append("validator: float64 robustness rows wrong")
    for r in rows:
        cell = CELLS[r["cell_index"]]
        exp_start = START_SEED_BASE + 100_000 * r["replicate"] \
            + 1_000 * r["cell_index"]
        exp_noise = NOISE_SEED_BASE + 100_000 * r["replicate"] \
            + 1_000 * r["cell_index"] + METHODS.index(r["method"])
        checks = [
            r["a"] == cell["a"], r["w"] == cell["w"], r["K"] == cell["K"],
            r["eta"] == cell["eta"], r["T"] == T_STEPS, r["G"] == G_ENS,
            r["dim"] == DIM,
            r["burnin"] == (0 if cell["cold"] else T_STEPS // 2),
            r["start_seed"] == exp_start, r["noise_seed"] == exp_noise,
            r["dtype_accum"] == "float64",
            r.get("precommit_sha256") == pre_sha,
            r["nonfinite"] == 0, r["max_sq_norm"] < NORM_CEILING,
            all(math.isfinite(v) for v in r["f_bar"]),
        ]
        if not all(checks):
            fail.append(f"validator: row ({r['cell_index']},{r['method']},"
                        f"{r['replicate']}) field check failed at position "
                        f"{checks.index(False)}")

    def arm(ci, m, field):
        return [(r["replicate"], r[field]) for r in acc
                if r["cell_index"] == ci and r["method"] == m]

    # ---- gates
    gates = {}
    for ci, cell in enumerate(CELLS):
        K = cell["K"]
        for m in METHODS:
            if m == "iid":
                continue
            tm = arm(ci, m, "transitions_kept")
            ti = arm(ci, "iid", "transitions_kept")
            em = [(rep, [sum(v) / len(v) for v in _chunks(vs, K)])
                  for rep, vs in tm]
            ei = [(rep, [sum(v) / len(v) for v in _chunks(vs, K)])
                  for rep, vs in ti]
            d, se, okg = four_sigma(em, ei)
            gates[f"escape-law c{ci} {m}"] = okg
            for fld, name in (("sec_x1", "E[x1^2]"),
                              ("sec_all_per_dim", "E[|x|^2]/d")):
                vm = [(rep, [sum(v) / len(v) for v in _chunks(vs, K)])
                      for rep, vs in arm(ci, m, fld)]
                vi = [(rep, [sum(v) / len(v) for v in _chunks(vs, K)])
                      for rep, vs in arm(ci, "iid", fld)]
                d, se, okg = four_sigma(vm, vi)
                gates[f"invariance {name} c{ci} {m}"] = okg
            d, se, okg = four_sigma(arm(ci, m, "f_bar"),
                                    arm(ci, "iid", "f_bar"))
            gates[f"occupancy-mean c{ci} {m}"] = okg
            # FPT gate with the registered censoring substitution
            fpt_i = [v for _, vs in arm(ci, "iid", "fpt") for v in vs]
            cens_frac = sum(1 for v in fpt_i if v < 0) / len(fpt_i)
            if cens_frac < 0.25:
                # Amended 2026-07-27 (see precommit amendment): the gate
                # statistic is the per-ensemble MEAN of uncensored FPTs.
                # Linear in the per-chain values, so its expectation is
                # invariant to within-ensemble dependence; the earlier
                # within-ensemble median shifts under common noise even
                # with identical per-chain marginals.
                def ens_mean(pairs, k):
                    out = []
                    for rep, vs in pairs:
                        means = []
                        for ch in _chunks(vs, k):
                            unc = [v for v in ch if v >= 0]
                            if unc:
                                means.append(sum(unc) / len(unc))
                        out.append((rep, means))
                    return out
                d, se, okg = four_sigma(ens_mean(arm(ci, m, "fpt"), K),
                                        ens_mean(arm(ci, "iid", "fpt"), K))
                gates[f"fpt-mean c{ci} {m}"] = okg
            else:
                def esc_frac(pairs, k):
                    return [(rep, [sum(1 for v in ch if v >= 0) / len(ch)
                                   for ch in _chunks(vs, k)])
                            for rep, vs in pairs]
                d, se, okg = four_sigma(esc_frac(arm(ci, m, "fpt"), K),
                                        esc_frac(arm(ci, "iid", "fpt"), K))
                gates[f"fpt-substituted-escfrac c{ci} {m}"] = okg
        h1, h2 = arm(ci, "iid", "half1_occ"), arm(ci, "iid", "half2_occ")
        d, se, okg = four_sigma(h1, h2)
        gates[f"stationarity-halves c{ci}"] = okg or cell["cold"]
        tr_iid = [v for _, vs in arm(ci, "iid", "transitions_kept")
                  for v in vs]
        mean_tr = sum(tr_iid) / len(tr_iid)
        frac_ge1 = sum(1 for v in tr_iid if v >= 1) / len(tr_iid)
        if not cell["cold"]:
            gates[f"floor-count c{ci}"] = mean_tr >= 3
            gates[f"floor-frac c{ci}"] = frac_ge1 >= 0.9

    # common positive control (secondary full-vector ratio > 1)
    control = {}
    for ci in range(len(CELLS)):
        pc = _trace_ratio(acc, ci, "common")
        control[ci] = pc["ratio"] if pc else None
        gates[f"common-control c{ci}"] = bool(pc and pc["ratio"] > 1.0)

    bad_gates = [k for k, v in gates.items() if not v]
    report = {"validator_failures": fail, "gate_failures": bad_gates,
              "gates_passed": len(gates) - len(bad_gates),
              "gates_total": len(gates), "cells": {}}

    # ---- metrics
    for ci, cell in enumerate(CELLS):
        c = {"name": cell["name"], "a": cell["a"], "w": cell["w"],
             "K": cell["K"], "eta": cell["eta"]}
        for m in ("zero_sum", "antithetic", "common"):
            pr = pooled_ratio(arm(ci, m, "f_bar"), arm(ci, "iid", "f_bar"),
                              REPS, REPS)
            c[f"occ_ratio_{m}"] = _fmt(pr)
            c[f"occ_band_{m}"] = band_word(pr)
            prh = pooled_ratio(arm(ci, m, "f_bar_hyst"),
                               arm(ci, "iid", "f_bar_hyst"), REPS, REPS)
            c[f"occ_hyst_ratio_{m}"] = _fmt(prh)
            tr = _trace_ratio(acc, ci, m)
            c[f"trace_full_{m}"] = _fmt(tr)
            c[f"trace_x1_{m}"] = _fmt(_trace_ratio(acc, ci, m, coords=[0]))
            c[f"trace_block_{m}"] = _fmt(
                _trace_ratio(acc, ci, m, coords=list(range(1, DIM))))
        for m in METHODS:
            rows_m = [r for r in acc if r["cell_index"] == ci
                      and r["method"] == m]
            p = sum(r["f_pool_sum"] for r in rows_m) \
                / sum(r["f_pool_n"] for r in rows_m)
            f2 = sum(r["f2_pool_sum"] for r in rows_m) \
                / sum(r["f_pool_n"] for r in rows_m)
            var_f = f2 - p * p
            K = cell["K"]
            if 0 < p < 1:
                rho = (K * var_f / (p * (1 - p)) - 1) / (K - 1)
                c[f"rho_{m}"] = round(rho, 4)
        stuck = {}
        for m in METHODS:
            fb = [v for _, vs in arm(ci, m, "f_bar") for v in vs]
            stuck[m] = round(sum(1 for v in fb if v < 0.1 or v > 0.9)
                             / len(fb), 4)
        c["stuck_frac"] = stuck
        fptq = {}
        for m in METHODS:
            fpts = sorted(v for r in acc if r["cell_index"] == ci
                          and r["method"] == m for v in r["fpt"] if v >= 0)
            if fpts:
                n = len(fpts)
                fptq[m] = [fpts[n // 4], fpts[n // 2], fpts[3 * n // 4]]
        c["fpt_pooled_quartiles"] = fptq
        if cell["name"] == "asym":
            fb = [v for _, vs in arm(ci, "iid", "f_bar") for v in vs]
            c["asym_occupancy_gap"] = round(
                sum(fb) / len(fb) - cell["w"], 4)
        if cell["cold"]:
            c["checkpoints"] = _cold_metrics(acc, ci)
        report["cells"][ci] = c

    if f64r:
        c2 = {}
        for m in ("zero_sum", "antithetic", "common"):
            r64 = [r for r in f64r if r["method"] == m]
            ri64 = [r for r in f64r if r["method"] == "iid"]
            pr = pooled_ratio([(0, r64[0]["f_bar"])], [(0, ri64[0]["f_bar"])],
                              [0], [0])
            c2[m] = _fmt(pr)
        report["float64_check_cell2_rep0"] = c2

    out = Path(args.out).with_suffix(".emitter.json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in
                      ("validator_failures", "gate_failures", "gates_passed",
                       "gates_total")}, indent=1))
    if fail or bad_gates:
        print("EMITTER: ABORT — validator or gate failure; "
              "no table is licensed.")
        sys.exit(1)
    print(f"EMITTER: PASS — full report at {out}")
    for ci, c in report["cells"].items():
        print(f"cell {ci} ({c['name']}): occ zs={c['occ_ratio_zero_sum']} "
              f"[{c['occ_band_zero_sum']}] anti={c['occ_ratio_antithetic']} "
              f"[{c['occ_band_antithetic']}] "
              f"common={c['occ_ratio_common']} [{c['occ_band_common']}] "
              f"rho_zs={c.get('rho_zero_sum')}")


def _chunks(flat, k):
    return [flat[i:i + k] for i in range(0, len(flat), k)]


def _fmt(pr):
    if pr is None:
        return None
    return {"ratio": round(pr["ratio"], 5),
            "ci": [round(pr["lo"], 5), round(pr["hi"], 5)],
            "g2": [round(pr["g2_m"], 2), round(pr["g2_iid"], 2)],
            "per_rep": [round(v, 5) for v in pr["per_rep"]]}


def _trace_ratio(acc, ci, m, coords=None):
    coords = coords or list(range(DIM))

    def per_ens(rows_m):
        out = []
        for r in rows_m:
            mv = torch.tensor(r["mean_vec"],
                              dtype=torch.float64).view(r["G"], r["dim"])
            out.append((r["replicate"],
                        mv[:, coords] if len(coords) > 1
                        else mv[:, coords[0]:coords[0] + 1]))
        return out

    rows_m = [r for r in acc if r["cell_index"] == ci and r["method"] == m]
    rows_i = [r for r in acc if r["cell_index"] == ci
              and r["method"] == "iid"]
    if not rows_m or not rows_i:
        return None

    def q_of(pairs):
        qs = []
        for _, mv in pairs:
            qs.append(((mv - mv.mean(dim=0, keepdim=True)) ** 2).sum(dim=1))
        return torch.cat(qs)
    qm, qi = q_of(per_ens(rows_m)), q_of(per_ens(rows_i))
    n = len(qm)
    if float(qi.mean()) == 0:
        return None
    ratio = float(qm.mean() / qi.mean())
    cv2m = float(qm.var() / qm.mean() ** 2)
    cv2i = float(qi.var() / qi.mean() ** 2)
    se = math.sqrt(cv2m / n + cv2i / n)
    per_rep = []
    for rm, ri_ in zip(per_ens(rows_m), per_ens(rows_i)):
        a_ = ((rm[1] - rm[1].mean(dim=0, keepdim=True)) ** 2).sum(dim=1)
        b_ = ((ri_[1] - ri_[1].mean(dim=0, keepdim=True)) ** 2).sum(dim=1)
        if float(b_.mean()) > 0:
            per_rep.append(float(a_.mean() / b_.mean()))
    return {"ratio": ratio, "lo": ratio * math.exp(-2 * se),
            "hi": ratio * math.exp(2 * se), "g2_m": cv2m - 2,
            "g2_iid": cv2i - 2, "per_rep": per_rep, "n": n}


def _cold_metrics(acc, ci):
    out = {}
    rows_i = {r["replicate"]: r for r in acc
              if r["cell_index"] == ci and r["method"] == "iid"}
    for t in [str(c) for c in CHECKPOINTS]:
        entry = {}
        iid_any = [rows_i[rep]["checkpoints"][t]["any_right_frac"]
                   for rep in REPS]
        entry["iid_any_right_frac"] = round(sum(iid_any) / 3, 4)
        for m in METHODS:
            rows_m = [r for r in acc if r["cell_index"] == ci
                      and r["method"] == m]
            entry[f"zero_escape_{m}"] = round(sum(
                r["checkpoints"][t]["zero_escape_frac"]
                for r in rows_m) / 3, 4)
        degenerate = entry["iid_any_right_frac"] < 0.10
        if not degenerate:
            vi = [(rep, rows_i[rep]["checkpoints"][t]["fbar_running"])
                  for rep in REPS]
            ti = torch.tensor([v for _, vs in vi for v in vs])
            if float(ti.var()) < 1e-6:
                degenerate = True
            else:
                for m in ("zero_sum", "antithetic", "common"):
                    vm = [(rep, r["checkpoints"][t]["fbar_running"])
                          for r in acc if r["cell_index"] == ci
                          and r["method"] == m for rep in [r["replicate"]]]
                    pr = pooled_ratio(vm, vi, REPS, REPS)
                    entry[f"ratio_{m}"] = _fmt(pr)
                    entry[f"band_{m}"] = band_word(pr)
        entry["degenerate"] = degenerate
        out[t] = entry
    return out


# ---------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--campaign", action="store_true")
    p.add_argument("--emit", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--burnin", type=int, default=20_000)
    p.add_argument("--ensembles", type=int, default=64)
    p.add_argument("--chains", type=int, default=8)
    p.add_argument("--dim", type=int, default=10)
    p.add_argument("--separations", type=float, nargs="+",
                   default=PILOT_SEPARATIONS)
    p.add_argument("--etas", type=float, nargs="+", default=PILOT_ETAS)
    p.add_argument("--no-asym", action="store_true")
    p.add_argument("--seed-base", type=int, default=SEED_BASE)
    p.add_argument("--precommit",
                   default="multimodal_bimodal_precommit.md")
    p.add_argument("--out", default="outputs/multimodal_pilot.jsonl")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    if args.selftest:
        sys.exit(0 if selftest(device) else 1)

    if args.emit:
        emit(args)
        return

    if args.campaign:
        if not Path(args.precommit).exists():
            sys.exit(f"refusing to run campaign without precommit file "
                     f"{args.precommit}")
        if not selftest(device):
            sys.exit("selftest failed — whole-campaign abort")
        T = 2_000 if args.quick else T_STEPS
        G = 16 if args.quick else G_ENS
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and not args.quick:
            sys.exit(f"refusing to overwrite {out}")
        prov = provenance(args.precommit)
        runs = [(ci, m, rep, torch.float32)
                for ci in range(len(CELLS)) for m in METHODS for rep in REPS]
        runs += [(2, m, 0, torch.float64) for m in METHODS]
        if args.quick:
            runs = [(0, m, 0, torch.float32) for m in METHODS] \
                + [(4, "zero_sum", 0, torch.float32)]
        with out.open("a") as fh:
            for ci, m, rep, dtype in runs:
                row = campaign_cell(ci, m, rep, device, dtype, T, G,
                                    quick=args.quick)
                row.update(prov)
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                print(f"c{ci} {CELLS[ci]['name']:>12} {m:>10} r{rep} "
                      f"{row['dtype_dynamics']}: "
                      f"{row['steps_per_sec']:.0f} st/s "
                      f"maxsq={row['max_sq_norm']:.0f} "
                      f"nonfinite={row['nonfinite']}")
        print(f"wrote {out}")
        return

    if not args.pilot:
        print("pick a mode: --pilot / --campaign / --emit / --selftest")
        sys.exit(2)

    steps = 2_000 if args.quick else args.steps
    burnin = 400 if args.quick else args.burnin
    cells = [(a, w) for a in args.separations for w in PILOT_WEIGHTS]
    if not args.no_asym:
        cells.append(PILOT_ASYM)
    if args.quick:
        cells = cells[:2]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.quick:
        sys.exit(f"refusing to overwrite {out}")
    prov = provenance()
    with out.open("a") as fh:
        for ci, (a, w) in enumerate(cells):
            for ei, eta in enumerate(args.etas):
                for mi, method in enumerate(METHODS):
                    seed = args.seed_base + 10_000 * ci + 100 * ei + mi
                    stats = run_cell(a, w, eta, steps, burnin,
                                     args.ensembles, args.chains, args.dim,
                                     method, seed, device)
                    row = {"status": "exploratory-pilot", "a": a, "w": w,
                           "eta": eta, "method": method, "steps": steps,
                           "burnin": burnin, "ensembles": args.ensembles,
                           "chains": args.chains, "dim": args.dim,
                           "seed": seed, **stats, **prov}
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    fh.flush()
                    print(f"a={a} w={w} eta={eta} {method:>10}: "
                          f"tr_mean={stats['transitions_mean']:>8} "
                          f"ge1={stats['frac_chains_ge1']:.2f} "
                          f"{stats['steps_per_sec']:.0f} steps/s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
