"""ZV-composition study: couplings x control-variate estimator columns.

Factorial {iid, antithetic, zero_sum} x {plain, ZV-1, ZV-2diag} on the
registered cells of campaigns/zv_composition_precommit.md (three BLR
cells, tprior-wdbc, hier-adult). Targets, step counts, eta, and zero-init
starts clone scripts/real_logistic_gpu.py / scripts/nonlogconcave_real.py
(cupy -> torch). Stats machinery (pooled ratio, bands, 4-sigma, emitter
abort) clones scripts/multimodal_bimodal.py.

Modes:
  --selftest   gradient FD checks (synthetic data, no files), coupling
               checks, ZV plumbing check on an internal Gaussian toy
               (plumbing only, NOT a campaign cell), determinism.
  --fit        dedicated iid fitting runs -> frozen ZV coefficients per
               cell (outputs/zv_frozen_<cell>.json). Precommit: frozen-fit
               is primary; the same map is applied to every arm.
  --pilot      EXPLORATORY stability/magnitude check; no scored metric;
               rows refused by the emitter.
  --campaign   the pre-registered campaign. Refuses to run without the
               precommit and all frozen-coefficient files (shas recorded
               in every row).
  --emit       validator + gates (K1 positive control, K2 floor, 4-sigma
               ZV-column mean gate) + contrasts A/B/I + bands. Aborts on
               any validator or gate failure; band words printed by code.

Seed blocks 96x,xxx,xxx (disjoint from all prior campaigns).
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

import numpy as np
import torch

METHODS = ["iid", "antithetic", "zero_sum"]
REPS = [0, 1, 2]
FIT_SEED_BASE = 960_000_000
NOISE_SEED_BASE = 961_000_000
PILOT_SEED_BASE = 963_000_000     # outside the 95x decade of prior campaigns
FIT_ENSEMBLES = 256               # review item B2: keep coefficient noise
FIT_STEPS_MULT = 4                # delta^2 well below the ZV residual
NORM_CEILING = 1e6
BANDS = [(0.5, "transfers"), (0.8, "partial"), (1.25, "parity"),
         (float("inf"), "hurts")]
ADULT_COLUMNS = 15
ADULT_EDU_COL = 3

CELLS = [
    {"name": "blr-wdbc",     "model": "blr",    "dataset": "wdbc",
     "lam": 0.1, "df": None, "eta": 0.08, "steps": 900, "burnin": 450,
     "reps": 256},
    {"name": "blr-spambase", "model": "blr",    "dataset": "spambase",
     "lam": 0.1, "df": None, "eta": 0.08, "steps": 900, "burnin": 450,
     "reps": 192},
    {"name": "blr-adult",    "model": "blr",    "dataset": "adult",
     "lam": 0.1, "df": None, "eta": 0.08, "steps": 700, "burnin": 350,
     "reps": 192},
    {"name": "tprior-wdbc",  "model": "tprior", "dataset": "wdbc",
     "lam": 0.1, "df": 3.0, "eta": 0.08, "steps": 900, "burnin": 450,
     "reps": 256},
    {"name": "hier-adult",   "model": "hier",   "dataset": "adult",
     "lam": 0.1, "df": None, "eta": 0.08, "steps": 700, "burnin": 350,
     "reps": 192},
]
K_CHAINS = 8

# K1 positive-control thresholds on the plain-column zs/iid O1 trace ratio
# (precommit "Gates"): BLR trio band ceiling; record-anchored intervals.
K1_BOUNDS = {"blr-wdbc": (0.0, 0.05), "blr-spambase": (0.0, 0.05),
             "blr-adult": (0.0, 0.05), "tprior-wdbc": (0.063, 0.81),
             "hier-adult": (0.009, 0.081)}


# ------------------------------------------------------------- provenance

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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES",
                                               "unset"),
        "python": sys.version.split()[0],
        **gpu_facts(),
    }
    if precommit_path:
        p["precommit_sha256"] = sha256_of(precommit_path)
    return p


# -------------------------------------------------------------- couplings

def make_noise(shape, method, generator, device, dtype=torch.float32):
    reps, chains, dim = shape

    def randn(*s):
        return torch.randn(*s, generator=generator, device=device,
                           dtype=dtype)
    if method == "iid":
        return randn(reps, chains, dim)
    if method == "zero_sum":
        z = randn(reps, chains, dim)
        z = z - z.mean(dim=1, keepdim=True)
        return z * math.sqrt(chains / (chains - 1))
    if method == "antithetic":
        if chains % 2:
            z = randn(reps, chains // 2, dim)
            extra = randn(reps, 1, dim)
            return torch.cat([z, -z, extra], dim=1)
        z = randn(reps, chains // 2, dim)
        return torch.cat([z, -z], dim=1)
    raise ValueError(f"unknown method {method}")


# ---------------------------------------------------------------- targets

def sigmoid(z):
    return torch.sigmoid(torch.clamp(z, -30.0, 30.0))


def add_intercept(x_np):
    ones = np.ones((x_np.shape[0], 1), dtype=np.float32)
    return np.concatenate([ones, x_np.astype(np.float32)], axis=1)


def load_blr(data_dir, name, device):
    data = np.load(Path(data_dir) / f"{name}.npz", allow_pickle=True)
    xt = torch.as_tensor(add_intercept(data["X_train"]), device=device)
    yt = torch.as_tensor(np.asarray(data["y_train"], dtype=np.float32),
                         device=device)
    xs = torch.as_tensor(add_intercept(data["X_test"]), device=device)
    return xt, yt, xs


def parse_adult_groups(raw_path):
    groups = []
    with Path(raw_path).open("r", encoding="utf-8") as f:
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


def load_hier(data_dir, raw_dir, device):
    data = np.load(Path(data_dir) / "adult.npz", allow_pickle=True)
    names = [str(v) for v in data["feature_names"]]
    keep = np.asarray([not v.startswith("education=") for v in names],
                      dtype=bool)
    x_np = add_intercept(data["X_train"][:, keep])
    group_idx, levels = parse_adult_groups(Path(raw_dir) / "adult.data")
    if len(group_idx) != x_np.shape[0]:
        raise RuntimeError(
            f"raw/npz row mismatch: {len(group_idx)} vs {x_np.shape[0]}")
    n_groups = len(levels)
    gmat_np = np.zeros((x_np.shape[0], n_groups), dtype=np.float32)
    gmat_np[np.arange(x_np.shape[0]), group_idx] = 1.0
    return (torch.as_tensor(x_np, device=device),
            torch.as_tensor(np.asarray(data["y_train"], dtype=np.float32),
                            device=device),
            torch.as_tensor(gmat_np, device=device), n_groups)


def blr_grad(flat, data_x, data_y, lam, df, s2, chunk):
    """-grad log posterior per unit == grad U; mean-likelihood scaling.

    df is None -> Gaussian prior lam (intercept unpenalized); df set ->
    t(df) prior of scale^2 s2 (intercept unpenalized). Clones
    real_logistic_gpu.logistic_grad / nonlogconcave_real.tprior_grad.
    """
    out = torch.empty_like(flat)
    mask = torch.ones(flat.shape[-1], device=flat.device)
    mask[0] = 0.0
    n = data_x.shape[0]
    for s in range(0, flat.shape[0], chunk):
        e = min(s + chunk, flat.shape[0])
        part = flat[s:e]
        probs = sigmoid(part @ data_x.T)
        g = (probs - data_y[None, :]) @ data_x / n
        if df is None:
            g = g + lam * part * mask[None, :]
        else:
            g = g + (df + 1.0) * part / (df * s2 + part * part) \
                * mask[None, :]
        out[s:e] = g
    return out


def hier_grad(flat, data_x, data_y, gmat, lam_beta, chunk):
    out = torch.empty_like(flat)
    n, db = data_x.shape
    n_groups = gmat.shape[1]
    mask = torch.ones(db, device=flat.device)
    mask[0] = 0.0
    for s in range(0, flat.shape[0], chunk):
        e = min(s + chunk, flat.shape[0])
        beta = flat[s:e, :db]
        u = flat[s:e, db:db + n_groups]
        rho = flat[s:e, db + n_groups]
        z = beta @ data_x.T + u @ gmat.T
        resid = sigmoid(z) - data_y[None, :]
        g_beta = resid @ data_x / n + lam_beta * beta * mask[None, :]
        inv_tau2 = torch.exp(-2.0 * torch.clamp(rho, -45.0, 45.0))
        g_u = resid @ gmat / n + u * inv_tau2[:, None]
        g_rho = -(u * u).sum(dim=1) * inv_tau2 + n_groups + rho
        out[s:e, :db] = g_beta
        out[s:e, db:db + n_groups] = g_u
        out[s:e, db + n_groups] = g_rho
    return out


class Target:
    """Bundles grad U, the O2 predictive observable, and dims for a cell.

    Data tensors are cached per dtype so float64 robustness rows matmul
    against float64 copies (torch does not promote dtypes in matmul; the
    design matrices are float32-SOURCED in every arm, so the robustness
    rows isolate dynamics/accumulation precision only — registered in the
    precommit's Float64 robustness note).
    """

    def __init__(self, cell, data_dir, raw_dir, device, chunk):
        self.cell = cell
        self.chunk = chunk
        self._cache = {}
        m = cell["model"]
        if m in ("blr", "tprior"):
            xt, yt, xs = load_blr(data_dir, cell["dataset"], device)
            self._base = (xt, yt, xs)
            self.dim = int(xt.shape[1])
            df = cell["df"]
            s2 = None if df is None else (df - 2.0) / (df * cell["lam"])
            self.args = (cell["lam"], df, s2)
        elif m == "hier":
            xt, yt, gmat, ng = load_hier(data_dir, raw_dir, device)
            self._base = (xt, yt, gmat)
            self.db = int(xt.shape[1])
            self.ng = ng
            self.dim = self.db + ng + 1
            self.args = (cell["lam"],)
        else:
            raise ValueError(m)

    def tensors(self, dtype):
        if dtype not in self._cache:
            self._cache[dtype] = tuple(t.to(dtype) for t in self._base)
        return self._cache[dtype]

    def grad(self, x):
        flat = x.reshape(-1, x.shape[-1])
        tt = self.tensors(x.dtype)
        if self.cell["model"] == "hier":
            g = hier_grad(flat, tt[0], tt[1], tt[2], self.args[0],
                          self.chunk)
        else:
            g = blr_grad(flat, tt[0], tt[1], *self.args, self.chunk)
        return g.reshape(x.shape)

    def o2(self, x):
        """Mean predictive probability per (ensemble, chain) state."""
        flat = x.reshape(-1, x.shape[-1])
        tt = self.tensors(x.dtype)
        outs = []
        if self.cell["model"] == "hier":
            xt, _, gmat = tt
            for s in range(0, flat.shape[0], self.chunk):
                e = min(s + self.chunk, flat.shape[0])
                beta = flat[s:e, :self.db]
                u = flat[s:e, self.db:self.db + self.ng]
                outs.append(sigmoid(beta @ xt.T + u @ gmat.T).mean(dim=1))
        else:
            o2_x = tt[2]
            for s in range(0, flat.shape[0], self.chunk):
                e = min(s + self.chunk, flat.shape[0])
                outs.append(sigmoid(flat[s:e] @ o2_x.T).mean(dim=1))
        return torch.cat(outs).reshape(x.shape[:-1])


# --------------------------------------------------- one run (any mode)

def run_arm(target, method, noise_seed, reps, steps, burnin, device,
            dtype, want_hash=True, want_ws=True):
    """One (cell, coupling) run. Zero-init starts (clones the accepted BLR
    harness; no start seed). Accumulates per-ensemble float64 means of the
    state, the score z = -grad (the same gradient the update just used --
    no extra evaluations), the diagonal Stein features 1 + x*z, and O2.

    want_ws additionally accumulates per-ensemble SAMPLE-LEVEL second
    moments (phi phi^T, phi f) so the literature-convention per-ensemble
    whole-sample ZV fit is computable in --emit (review round 1 B5/B3): the
    registered fairness robustness line. Solved batched at end of run.
    """
    d = target.dim
    p2 = 2 * d
    gen = torch.Generator(device=device)
    gen.manual_seed(noise_seed)
    x = torch.zeros(reps, K_CHAINS, d, device=device, dtype=dtype)
    f64 = torch.float64
    acc = {k: torch.zeros(reps, d, device=device, dtype=f64)
           for k in ("x", "z", "w")}
    acc_o2 = torch.zeros(reps, device=device, dtype=f64)
    if want_ws:
        s_pp = torch.zeros(reps, p2, p2, device=device, dtype=f64)
        s_pf = torch.zeros(reps, p2, d + 1, device=device, dtype=f64)
        s_phi = torch.zeros(reps, p2, device=device, dtype=f64)
        s_f = torch.zeros(reps, d + 1, device=device, dtype=f64)
    root = math.sqrt(2.0 * target.cell["eta"])
    eta = target.cell["eta"]
    state_hash = None
    sq_max, nonfinite, kept = 0.0, 0, 0
    t0 = time.time()
    for t in range(steps):
        g = target.grad(x)
        if t >= burnin:
            z = -g
            o2v = target.o2(x)
            acc["x"] += x.mean(dim=1).to(f64)
            acc["z"] += z.mean(dim=1).to(f64)
            acc["w"] += (1.0 + x * z).mean(dim=1).to(f64)
            acc_o2 += o2v.mean(dim=1).to(f64)
            if want_ws:
                phi = torch.cat([z, 1.0 + x * z], dim=-1).to(f64)
                fob = torch.cat([x.to(f64), o2v.to(f64)[..., None]],
                                dim=-1)
                s_pp += torch.einsum("gkp,gkq->gpq", phi, phi)
                s_pf += torch.einsum("gkp,gkj->gpj", phi, fob)
                s_phi += phi.sum(dim=1)
                s_f += fob.sum(dim=1)
            kept += 1
        x = x - eta * g + root * make_noise((reps, K_CHAINS, d), method,
                                            gen, device, dtype)
        if t == 49 and want_hash:
            state_hash = hashlib.sha256(
                x.detach().to("cpu", torch.float64).numpy().tobytes()
            ).hexdigest()[:16]
        if (t + 1) % max(1, steps // 4) == 0:
            sq_max = max(sq_max, float((x * x).sum(dim=-1).max()))
            nonfinite += int((~torch.isfinite(x)).sum())
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    for k in acc:
        acc[k] /= kept
    out = {"mean_x": acc["x"], "mean_z": acc["z"], "mean_w": acc["w"],
           "mean_o2": acc_o2 / kept, "kept": kept,
           "state_hash_50": state_hash, "max_sq_norm": round(sq_max, 2),
           "nonfinite": nonfinite, "wall_sec": round(dt, 2),
           "steps_per_sec": round(steps / dt, 1)}
    if want_ws:
        n = kept * K_CHAINS
        mphi, mf = s_phi / n, s_f / n
        cpp = s_pp / n - torch.einsum("gp,gq->gpq", mphi, mphi)
        cpf = s_pf / n - torch.einsum("gp,gj->gpj", mphi, mf)
        for tag, p in (("ws1", d), ("ws2", p2)):
            gram = cpp[:, :p, :p].clone()
            ridge = torch.clamp(
                1e-8 * torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)
                / p, min=1e-30)
            eye = torch.eye(p, device=device, dtype=f64)[None]
            # Escalating-ridge retries, never lstsq: the minimum-norm
            # solution on a singular ensemble is all zeros, silently
            # degrading the WS column to the plain column (review round 2),
            # and CUDA gels is undefined for rank-deficient input.
            b, esc_used = None, 0.0
            for esc in (1.0, 1e4, 1e8):
                try:
                    b = torch.linalg.solve(
                        gram + esc * ridge[:, None, None] * eye,
                        cpf[:, :p, :])                   # (G,p,d+1)
                    esc_used = esc
                    break
                except Exception:
                    continue
            if b is None:
                raise RuntimeError(
                    f"{tag} solve failed after ridge escalation")
            adj = mf - torch.einsum("gp,gpj->gj", mphi[:, :p], b)
            out[tag] = adj                               # (G,d+1)
            out[f"{tag}_ridge_esc"] = esc_used
    return out


def rnd(t, sig=8):
    return [float(f"{v:.{sig}g}") for v in t.flatten().tolist()]


# ------------------------------------------------------------- fit mode

def fit_cell(ci, target, device, dtype, out_dir, prov, quick=False):
    """Dedicated iid fitting run -> frozen ZV coefficients (precommit
    'FROZEN COEFFICIENTS'). Pooled float64 LS over all kept samples of all
    fitting chains; ridge 1e-8*tr(G)/p. Also the H1/H2 time-half holdout
    R^2 diagnostics (disclosure; frozen coefficients use the FULL run).

    Per-sample regression: response f(x) in {x_j} u {O2}, features
    phi1 = z, phi2 = (z, 1 + x*z). Accumulated as second moments.
    """
    cell = target.cell
    d = target.dim
    steps = (cell["steps"] * FIT_STEPS_MULT) if not quick \
        else max(40, cell["steps"] // 4)
    burnin = steps // 2
    gen = torch.Generator(device=device)
    gen.manual_seed(FIT_SEED_BASE + 1_000 * ci)
    reps = FIT_ENSEMBLES if not quick else 4
    x = torch.zeros(reps, K_CHAINS, d, device=device, dtype=dtype)
    f64 = torch.float64
    p2 = 2 * d
    # sufficient stats for halves H1/H2 of the kept window
    stats = [dict(n=0,
                  s_phi=torch.zeros(p2, device=device, dtype=f64),
                  s_f=torch.zeros(d + 1, device=device, dtype=f64),
                  s_pp=torch.zeros(p2, p2, device=device, dtype=f64),
                  s_pf=torch.zeros(p2, d + 1, device=device, dtype=f64),
                  s_ff=torch.zeros(d + 1, device=device, dtype=f64))
             for _ in range(2)]
    kept_total = steps - burnin
    half_mark = burnin + kept_total // 2
    root = math.sqrt(2.0 * cell["eta"])
    eta = cell["eta"]
    for t in range(steps):
        g = target.grad(x)
        if t >= burnin:
            z = -g
            phi = torch.cat([z, 1.0 + x * z], dim=-1)     # (reps,K,2d)
            fobs = torch.cat([x, target.o2(x)[..., None]], dim=-1)
            pf = phi.reshape(-1, p2).to(f64)
            ff = fobs.reshape(-1, d + 1).to(f64)
            st = stats[0] if t < half_mark else stats[1]
            st["n"] += pf.shape[0]
            st["s_phi"] += pf.sum(dim=0)
            st["s_f"] += ff.sum(dim=0)
            st["s_pp"] += pf.T @ pf
            st["s_pf"] += pf.T @ ff
            st["s_ff"] += (ff * ff).sum(dim=0)
        x = x - eta * g + root * make_noise((reps, K_CHAINS, d), "iid",
                                            gen, device, dtype)

    def moments(st):
        n = st["n"]
        mp = st["s_phi"] / n
        mf = st["s_f"] / n
        cpp = st["s_pp"] / n - torch.outer(mp, mp)
        cpf = st["s_pf"] / n - torch.outer(mp, mf)
        vf = st["s_ff"] / n - mf * mf
        return cpp, cpf, vf

    def solve(cpp, cpf, p):
        """Ridge solve + conditioning facts of the UNRIDGED Gram (gate
        round-1 B3: cond of the ridged matrix is bounded ~1e8*p by the
        ridge rule itself, so K2b must threshold the unridged one)."""
        gram = cpp[:p, :p]
        ridge = max(1e-8 * float(torch.diagonal(gram).sum()) / p, 1e-30)
        eye_p = torch.eye(p, device=device, dtype=f64)
        coef = None
        for esc in (1.0, 1e4, 1e8):     # escalating ridge, never lstsq
            try:
                coef = torch.linalg.solve(gram + esc * ridge * eye_p,
                                          cpf[:p])
                break
            except Exception:
                continue
        if coef is None:
            raise RuntimeError("fit solve failed after ridge escalation")
        ev = torch.linalg.eigvalsh(gram)
        l_min, l_max = float(ev[0]), float(ev[-1])
        cond_unridged = l_max / max(l_min, 1e-300)
        return coef, {"cond_unridged": cond_unridged, "l_min": l_min,
                      "ridge": ridge,
                      "ridge_loadbearing": bool(ridge > 1e-3
                                                * max(l_min, 0.0))}

    # frozen coefficients from the FULL run (H1+H2 pooled)
    pooled = {k: stats[0][k] + stats[1][k] if k != "n"
              else stats[0]["n"] + stats[1]["n"] for k in stats[0]}
    cpp, cpf, _ = moments(pooled)
    coef1, cfacts1 = solve(cpp, cpf, d)
    coef2, cfacts2 = solve(cpp, cpf, p2)
    # H1/H2 half-fits: holdout R^2 diagnostics AND the emit-side
    # fit-noise gate columns (review round 1 B2: coefficient noise inflates
    # only the iid ZV column; tri-fit disagreement flags it).
    cpp1, cpf1, _ = moments(stats[0])
    cpp2, cpf2, vf2 = moments(stats[1])
    halves, r2 = {}, {}
    for tag, p in (("deg1", d), ("deg2", p2)):
        bh1, _ = solve(cpp1, cpf1, p)
        bh2, _ = solve(cpp2, cpf2, p)
        halves[tag] = (bh1, bh2)
        res = vf2 - 2.0 * (bh1 * cpf2[:p]).sum(dim=0) \
            + (bh1 * (cpp2[:p, :p] @ bh1)).sum(dim=0)
        r2v = 1.0 - res / torch.clamp(vf2, min=1e-300)
        r2[tag] = {"o1_mean": round(float(r2v[:d].mean()), 6),
                   "o1_min": round(float(r2v[:d].min()), 6),
                   "o2": round(float(r2v[d]), 6)}
    out = {
        "cell_index": ci, "cell_name": cell["name"], "dim": d,
        "fit_steps": steps, "fit_burnin": burnin, "fit_ensembles": reps,
        "fit_seed": FIT_SEED_BASE + 1_000 * ci, "chains": K_CHAINS,
        "eta": cell["eta"], "ridge_rule": "1e-8*tr(G)/p",
        "cond_deg1": cfacts1, "cond_deg2": cfacts2, "r2_holdout": r2,
        "coef_deg1": [rnd(coef1[:, j], 12) for j in range(d + 1)],
        "coef_deg2": [rnd(coef2[:, j], 12) for j in range(d + 1)],
        "coef_deg1_h1": [rnd(halves["deg1"][0][:, j], 12)
                         for j in range(d + 1)],
        "coef_deg1_h2": [rnd(halves["deg1"][1][:, j], 12)
                         for j in range(d + 1)],
        "coef_deg2_h1": [rnd(halves["deg2"][0][:, j], 12)
                         for j in range(d + 1)],
        "coef_deg2_h2": [rnd(halves["deg2"][1][:, j], 12)
                         for j in range(d + 1)],
        "status": "fitting-run" + ("-quick" if quick else ""), **prov,
    }
    fname = (f"zv_frozen_quick_{cell['name']}.json" if quick
             else f"zv_frozen_{cell['name']}.json")
    path = Path(out_dir) / fname
    if path.exists() and not quick:
        sys.exit(f"refusing to overwrite frozen coefficients {path} "
                 f"(review round 1 B6: the frozen map binds at lock)")
    path.write_text(json.dumps(out, sort_keys=True))
    print(f"fit {cell['name']}: cond_unridged deg1="
          f"{cfacts1['cond_unridged']:.3g} deg2="
          f"{cfacts2['cond_unridged']:.3g} "
          f"R2 deg1 o1_mean={r2['deg1']['o1_mean']} o2={r2['deg1']['o2']} "
          f"deg2 o1_mean={r2['deg2']['o1_mean']} o2={r2['deg2']['o2']}")
    return path


def load_frozen(out_dir, cell, device, ci):
    """Load + validate a frozen-coefficient file. Refuses quick artifacts,
    wrong fit parameters, and wrong shapes (review rounds 1-2: nothing but
    the registered fitting run may become the frozen map, and the seed
    check is anchored to the CELLS position, not the file's own claim)."""
    path = Path(out_dir) / f"zv_frozen_{cell['name']}.json"
    data = json.loads(path.read_text())
    d = data["dim"]
    checks = {
        "status": data.get("status") == "fitting-run",
        "cell_index": data.get("cell_index") == ci,
        "fit_ensembles": data.get("fit_ensembles") == FIT_ENSEMBLES,
        "fit_steps": data.get("fit_steps")
        == FIT_STEPS_MULT * cell["steps"],
        "fit_seed": data.get("fit_seed") == FIT_SEED_BASE + 1_000 * ci,
        "chains": data.get("chains") == K_CHAINS,
        "eta": data.get("eta") == cell["eta"],
        "cell_name": data.get("cell_name") == cell["name"],
        "shape_deg1": len(data["coef_deg1"]) == d + 1
        and all(len(c) == d for c in data["coef_deg1"]),
        "shape_deg2": len(data["coef_deg2"]) == d + 1
        and all(len(c) == 2 * d for c in data["coef_deg2"]),
        "shape_halves": all(
            len(data[k]) == d + 1
            and all(len(c) == (d if "deg1" in k else 2 * d)
                    for c in data[k])
            for k in ("coef_deg1_h1", "coef_deg1_h2",
                      "coef_deg2_h1", "coef_deg2_h2")),
    }
    bad = [k for k, v in checks.items() if not v]
    if bad:
        sys.exit(f"frozen-coefficient file {path} fails registered fit "
                 f"checks: {bad}")

    def mat(key):
        return torch.tensor(np.asarray(data[key], dtype=np.float64).T,
                            device=device)
    return {"deg1": mat("coef_deg1"), "deg2": mat("coef_deg2"),
            "deg1_h1": mat("coef_deg1_h1"), "deg1_h2": mat("coef_deg1_h2"),
            "deg2_h1": mat("coef_deg2_h1"), "deg2_h2": mat("coef_deg2_h2"),
            "dim": d, "sha": sha256_of(path),
            "cond_deg1": data["cond_deg1"], "cond_deg2": data["cond_deg2"],
            "r2": data["r2_holdout"], "path": str(path)}


# -------------------------------------------------------- campaign mode

def campaign_row(ci, target, method, rep, device, dtype, frozen, prov):
    cell = target.cell
    noise_seed = NOISE_SEED_BASE + 100_000 * rep + 1_000 * ci \
        + METHODS.index(method)
    r = run_arm(target, method, noise_seed, cell["reps"], cell["steps"],
                cell["burnin"], device, dtype)
    row = {
        "status": ("float64-robustness" if dtype == torch.float64
                   else "accepted"),
        "cell_index": ci, "cell_name": cell["name"], "model": cell["model"],
        "dataset": cell["dataset"], "K": K_CHAINS, "eta": cell["eta"],
        "steps": cell["steps"], "burnin": cell["burnin"],
        "G": cell["reps"], "dim": target.dim, "method": method,
        "replicate": rep, "noise_seed": noise_seed,
        "dtype_dynamics": str(dtype).split(".")[-1],
        "dtype_accum": "float64", "frozen_sha": frozen["sha"],
        "state_hash_50": r["state_hash_50"],
        "max_sq_norm": r["max_sq_norm"], "nonfinite": r["nonfinite"],
        "kept": r["kept"], "wall_sec": r["wall_sec"],
        "steps_per_sec": r["steps_per_sec"],
        "mean_x": rnd(r["mean_x"]), "mean_z": rnd(r["mean_z"]),
        "mean_w": rnd(r["mean_w"]), "mean_o2": rnd(r["mean_o2"]),
        "ws1_o1": rnd(r["ws1"][:, :target.dim]),
        "ws1_o2": rnd(r["ws1"][:, target.dim]),
        "ws2_o1": rnd(r["ws2"][:, :target.dim]),
        "ws2_o2": rnd(r["ws2"][:, target.dim]),
        "ws1_ridge_esc": r["ws1_ridge_esc"],
        "ws2_ridge_esc": r["ws2_ridge_esc"], **prov,
    }
    return row


# ----------------------------------------------------------- estimators

def columns_from_row(row, frozen, device):
    """Per-ensemble estimator columns from stored per-ensemble stats.
    Frozen map: identical across arms. Returns dict of (G,dim) / (G,)
    float64 tensors keyed (obs, column)."""
    g, d = row["G"], row["dim"]
    mx = torch.tensor(row["mean_x"], dtype=torch.float64,
                      device=device).view(g, d)
    mz = torch.tensor(row["mean_z"], dtype=torch.float64,
                      device=device).view(g, d)
    mw = torch.tensor(row["mean_w"], dtype=torch.float64,
                      device=device).view(g, d)
    mo2 = torch.tensor(row["mean_o2"], dtype=torch.float64, device=device)
    phi1, phi2 = mz, torch.cat([mz, mw], dim=1)
    cols = {("o1", "plain"): mx, ("o2", "plain"): mo2}
    for tag, phi, key in (("zv1", phi1, "deg1"), ("zv2", phi2, "deg2"),
                          ("zv1h1", phi1, "deg1_h1"),
                          ("zv1h2", phi1, "deg1_h2"),
                          ("zv2h1", phi2, "deg2_h1"),
                          ("zv2h2", phi2, "deg2_h2")):
        c = frozen[key]
        cols[("o1", tag)] = mx - phi @ c[:, :d]
        cols[("o2", tag)] = mo2 - phi @ c[:, d]
    for tag in ("ws1", "ws2"):
        cols[("o1", tag)] = torch.tensor(row[f"{tag}_o1"],
                                         dtype=torch.float64,
                                         device=device).view(g, d)
        cols[("o2", tag)] = torch.tensor(row[f"{tag}_o2"],
                                         dtype=torch.float64,
                                         device=device)
    cols[("z", "plain")] = mz          # score-invariance abort gate
    return cols


# -------------------------------------------- pooled ratios (clone)

def pooled_ratio(vals_m, vals_iid):
    """Clone of multimodal_bimodal.pooled_ratio generalized to vector
    summaries: q = ||per-replicate-centered value||^2 per ensemble."""
    def q_of(pairs):
        qs = []
        for _, v in pairs:
            v2 = v if v.dim() == 2 else v[:, None]
            qs.append(((v2 - v2.mean(dim=0, keepdim=True)) ** 2)
                      .sum(dim=1))
        return torch.cat(qs)
    qm, qi = q_of(vals_m), q_of(vals_iid)
    n = len(qm)
    dof_scale = n / (n - len(vals_m))
    s2m = float(qm.mean() * dof_scale)
    s2i = float(qi.mean() * dof_scale)
    if s2i == 0:
        return None
    ratio = s2m / s2i
    cv2m = float(qm.var() / qm.mean() ** 2) if qm.mean() > 0 else 0.0
    cv2i = float(qi.var() / qi.mean() ** 2) if qi.mean() > 0 else 0.0
    se_log = math.sqrt(cv2m / n + cv2i / n)
    per_rep = []
    for (_, vm), (_, vi) in zip(vals_m, vals_iid):
        vm2 = vm if vm.dim() == 2 else vm[:, None]
        vi2 = vi if vi.dim() == 2 else vi[:, None]
        a = ((vm2 - vm2.mean(dim=0, keepdim=True)) ** 2).sum(dim=1)
        b = ((vi2 - vi2.mean(dim=0, keepdim=True)) ** 2).sum(dim=1)
        if float(b.mean()) > 0:
            per_rep.append(float(a.mean() / b.mean()))
    return {"ratio": ratio, "lo": ratio * math.exp(-2 * se_log),
            "hi": ratio * math.exp(2 * se_log), "se_log": se_log,
            "g2_m": cv2m - 2, "g2_iid": cv2i - 2, "per_rep": per_rep,
            "n": n, "abs_m": s2m, "abs_iid": s2i}


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
    if len({band_of(r) for r in pr["per_rep"]}) > 1:
        return "unresolved"
    return b_point


def four_sigma_cols(vals_m, vals_iid, thresh=4.0):
    """Max over coordinates of |mean diff|/se (scalar obs: one coord).
    Returns (zmax, argmax coordinate, ok). ABORT-tier invariance gates
    pass thresh=5.0 (registered: ~1334 max-tests across the campaign put
    the spurious whole-campaign abort near 1e-3 at 5 sigma, vs 3-5% at
    4 sigma, which the precommit accepted for DEMOTE only)."""
    tm = torch.cat([v if v.dim() == 2 else v[:, None] for _, v in vals_m])
    ti = torch.cat([v if v.dim() == 2 else v[:, None] for _, v in vals_iid])
    se = torch.sqrt(tm.var(dim=0) / tm.shape[0]
                    + ti.var(dim=0) / ti.shape[0])
    zs = (tm.mean(dim=0) - ti.mean(dim=0)).abs() \
        / torch.clamp(se, min=1e-300)
    zmax = float(zs.max())
    return zmax, int(zs.argmax()), zmax <= thresh


# ------------------------------------------------------------- emitter

def emit(args, device):
    rows = [json.loads(l) for l in open(args.out)]
    acc = [r for r in rows if r["status"] == "accepted"]
    f64r = [r for r in rows if r["status"] == "float64-robustness"]
    stray = [r["status"] for r in rows
             if r["status"] not in ("accepted", "float64-robustness")]
    # Post-lock addenda change the living file's hash; --precommit-sha
    # pins the locked blob (review round 1 A8/A11).
    pre_sha = args.precommit_sha or sha256_of(args.precommit)
    frozen = {c["name"]: load_frozen(args.frozen_dir, c, device, ci)
              for ci, c in enumerate(CELLS)}
    fail = []
    if stray:
        fail.append(f"validator: stray statuses in campaign file {stray}")

    want = {(ci, m, rep) for ci in range(len(CELLS)) for m in METHODS
            for rep in REPS}
    got = {(r["cell_index"], r["method"], r["replicate"]) for r in acc}
    if got != want:
        fail.append(f"validator: inventory (missing {sorted(want-got)[:4]},"
                    f" extra {sorted(got-want)[:4]})")
    if {(r["cell_index"], r["method"]) for r in f64r} != \
            {(ci, m) for ci in range(len(CELLS)) for m in METHODS} \
            or any(r["replicate"] != 0 for r in f64r):
        fail.append("validator: float64 robustness rows wrong")
    # Cardinality + duplicate detection (review round 2: sets hide a shard
    # concatenated twice, which would shrink every CI by sqrt(2)).
    keys_all = [(r["cell_index"], r["method"], r["replicate"], r["status"])
                for r in acc + f64r]
    if len(keys_all) != len(set(keys_all)):
        fail.append("validator: duplicate rows (shard concatenated "
                    "twice?)")
    if len(acc) != 3 * len(CELLS) * len(METHODS) \
            or len(f64r) != len(CELLS) * len(METHODS):
        fail.append(f"validator: row cardinality acc={len(acc)} "
                    f"f64={len(f64r)}")
    vec_fields = ("mean_x", "mean_z", "mean_w", "mean_o2",
                  "ws1_o1", "ws1_o2", "ws2_o1", "ws2_o2")
    for r in acc + f64r:
        cell = CELLS[r["cell_index"]]
        exp_seed = NOISE_SEED_BASE + 100_000 * r["replicate"] \
            + 1_000 * r["cell_index"] + METHODS.index(r["method"])
        exp_dt = ("float64" if r["status"] == "float64-robustness"
                  else "float32")
        checks = [
            r["cell_name"] == cell["name"], r["K"] == K_CHAINS,
            r["eta"] == cell["eta"], r["steps"] == cell["steps"],
            r["burnin"] == cell["burnin"], r["G"] == cell["reps"],
            r["noise_seed"] == exp_seed, r["dtype_accum"] == "float64",
            r["dtype_dynamics"] == exp_dt,
            r["kept"] == cell["steps"] - cell["burnin"],
            r.get("precommit_sha256") == pre_sha,
            r.get("frozen_sha") == frozen[cell["name"]]["sha"],
            r["dim"] == frozen[cell["name"]]["dim"],
            r["nonfinite"] == 0, r["max_sq_norm"] < NORM_CEILING,
            all(f in r for f in vec_fields),
            all(math.isfinite(v) for f in vec_fields if f in r
                for v in r[f]),
        ]
        if not all(checks):
            fail.append(f"validator: row ({r['cell_index']},{r['method']},"
                        f"{r['replicate']},{r['status']}) failed at "
                        f"position {checks.index(False)}")
    hosts = {r["hostname"] for r in acc + f64r}
    if len(hosts) > 1:
        fail.append(f"validator: one-machine rule violated {sorted(hosts)}")
    # Bail BEFORE the gate loop: a malformed file must produce the
    # structured abort report, never a bare traceback from
    # columns_from_row / torch.cat on missing rows (review round 2).
    if fail:
        out = Path(args.out).with_suffix(".emitter.json")
        out.write_text(json.dumps({"validator_failures": fail},
                                  indent=1, sort_keys=True))
        print(json.dumps({"validator_failures": fail}, indent=1))
        print("EMITTER: ABORT — validator failure; no table is licensed.")
        sys.exit(1)

    def arm_cols(rows_src, ci, m):
        out = []
        for r in sorted(rows_src, key=lambda r: r["replicate"]):
            if r["cell_index"] == ci and r["method"] == m:
                out.append((r["replicate"],
                            columns_from_row(r, frozen[r["cell_name"]],
                                             device)))
        return out

    CEILING = K_CHAINS / (K_CHAINS - 1)          # 8/7 ~ 1.1429
    gates, report_cells = {}, {}
    for ci, cell in enumerate(CELLS):
        name = cell["name"]
        cols = {m: arm_cols(acc, ci, m) for m in METHODS}
        c64 = {m: arm_cols(f64r, ci, m) for m in METHODS}
        c32r0 = {m: [(rr_, cc) for rr_, cc in cols[m] if rr_ == 0]
                 for m in METHODS}          # rep-0 f32, K2's registered arm

        def series(m, obs, col):
            return [(rep, cc[(obs, col)]) for rep, cc in cols[m]]

        def vvar(m, obs, col, src=None):
            """Pooled across-ensemble variance of one estimator column."""
            s = ([(rep, cc[(obs, col)]) for rep, cc in src[m]]
                 if src else series(m, obs, col))
            pr = pooled_ratio(s, s)
            return pr["abs_m"] if pr else 0.0

        # K1 positive control on the plain column (ABORT on breach)
        pr_k1 = pooled_ratio(series("zero_sum", "o1", "plain"),
                             series("iid", "o1", "plain"))
        lo, hi = K1_BOUNDS[name]
        gates[f"K1-positive-control {name}"] = bool(
            pr_k1 and lo <= pr_k1["ratio"] <= hi)

        # Plain-column and score marginal-invariance gates (ABORT: a
        # breach here is an implementation bug, never a finding). 5-sigma
        # threshold; magnitude and coordinate recorded for the audit.
        inv_z = {}
        for m in ("zero_sum", "antithetic"):
            for obs, col in (("o1", "plain"), ("o2", "plain"),
                             ("z", "plain")):
                z, ji, okg = four_sigma_cols(series(m, obs, col),
                                            series("iid", obs, col),
                                            thresh=5.0)
                gates[f"invariance {name} {m} {obs}"] = okg
                inv_z[f"{m}/{obs}"] = [round(z, 2), ji]

        # 4-sigma ZV-1 column mean gate: breach DEMOTES the cell to
        # disclosed-with-caveat (precommit), it does not abort.
        demoted = []
        for m in ("zero_sum", "antithetic"):
            for obs in ("o1", "o2"):
                z, ji, okg = four_sigma_cols(series(m, obs, "zv1"),
                                            series("iid", obs, "zv1"))
                if not okg:
                    demoted.append(f"{m}/{obs} zmax={z:.2f} coord={ji}")

        fz = frozen[name]
        entry = {"name": name, "k1_plain_ratio": _fmt(pr_k1),
                 "demoted_4sigma": demoted, "invariance_z": inv_z,
                 "r2_holdout": fz["r2"],
                 "cond": {"deg1": fz["cond_deg1"],
                          "deg2": fz["cond_deg2"]}}
        for obs in ("o1", "o2"):
            plain = pooled_ratio(series("zero_sum", obs, "plain"),
                                 series("iid", obs, "plain"))
            for deg, col in (("deg1", "zv1"), ("deg2", "zv2")):
                key = f"{obs}_{deg}"
                a = pooled_ratio(series("zero_sum", obs, col),
                                 series("iid", obs, col))
                anti = pooled_ratio(series("antithetic", obs, col),
                                    series("iid", obs, col))
                entry[f"A_{key}"] = _fmt(a)
                entry[f"A_band_{key}"] = band_word(a)
                entry[f"A_anti_{key}"] = _fmt(anti)
                entry[f"A_anti_band_{key}"] = band_word(anti)
                for m in METHODS:
                    b = pooled_ratio(series(m, obs, col),
                                     series(m, obs, "plain"))
                    entry[f"B_{m}_{key}"] = _fmt(b)
                if a and plain:
                    i_val = math.log(a["ratio"]) - math.log(plain["ratio"])
                    i_se = math.sqrt(a["se_log"] ** 2
                                     + plain["se_log"] ** 2)
                    entry[f"I_{key}"] = {
                        "log_interaction": round(i_val, 4),
                        "ci": [round(i_val - 2 * i_se, 4),
                               round(i_val + 2 * i_se, 4)]}
                # Degree-law ceiling (gate rounds 1-2): the CI criterion
                # governs consistency (CI intersects (1, K/(K-1)]); a
                # point above the ceiling with CI reaching below it is a
                # DESCRIPTOR, not a different verdict. The diagnostic
                # ABORT tests at 4 sigma: at 2 sigma a true A sitting at
                # the ceiling -- exactly P-R1's prediction -- would
                # spuriously kill the whole table 15-27% of the time; a
                # real bug or floor sits many sigma out, so 4 loses no
                # power (registered arithmetic in the precommit).
                if a:
                    lo4 = a["ratio"] * math.exp(-4 * a["se_log"])
                    if lo4 > CEILING:
                        entry[f"ceiling_{key}"] = "VIOLATION-diagnostic"
                        gates[f"ceiling-diagnostic {name} {key}"] = False
                    elif a["lo"] > CEILING:
                        entry[f"ceiling_{key}"] = "exceeds-2sigma"
                        # the printed band word must not read "parity"
                        # beside an inconsistent-at-2-sigma ceiling label
                        entry[f"A_band_{key}"] = "exceeds-ceiling-2sigma"
                    elif a["hi"] > 1.0:
                        entry[f"ceiling_{key}"] = (
                            "consistent-point-above"
                            if a["ratio"] > CEILING else "consistent")
                    else:
                        entry[f"ceiling_{key}"] = "below-parity"
                # K2 floor rule, ALL arms (review round 1 A1): rep0 f32 vs
                # f64 column variance within 2x, else floor-limited.
                floor_m = []
                for m in METHODS:
                    if not c64[m] or not c32r0[m]:
                        continue    # inventory failure already recorded
                    v32 = vvar(m, obs, col, src=c32r0)
                    v64 = vvar(m, obs, col, src=c64)
                    if not (0.5 <= (v32 / max(v64, 1e-300)) <= 2.0):
                        floor_m.append(m)
                # K2 second trigger: unresolved with CI spanning a
                # decade, evaluated per contrast.
                decade = bool(a and band_word(a) == "unresolved"
                              and a["hi"] / max(a["lo"], 1e-300) > 10)
                decade_anti = bool(anti and band_word(anti) == "unresolved"
                                   and anti["hi"]
                                   / max(anti["lo"], 1e-300) > 10)
                entry[f"floor_methods_{key}"] = floor_m
                if decade or {"iid", "zero_sum"} & set(floor_m):
                    entry[f"A_band_{key}"] = "unresolved-at-floor"
                if decade_anti or {"iid", "antithetic"} & set(floor_m):
                    entry[f"A_anti_band_{key}"] = "unresolved-at-floor"
                # Fit-noise gate (review round 1 B2): columns under the two
                # half-fits must agree with the frozen fit.
                if col in ("zv1", "zv2"):
                    variants = [col, f"{col}h1", f"{col}h2"]
                    vs = [vvar("iid", obs, v) for v in variants]
                    bands = set()
                    for v in variants:
                        av = pooled_ratio(series("zero_sum", obs, v),
                                          series("iid", obs, v))
                        bands.add(band_word(av))
                    spread = (max(vs) / max(min(vs), 1e-300)
                              if min(vs) > 0 else float("inf"))
                    entry[f"fitnoise_spread_{key}"] = round(spread, 3)
                    if spread > 1.5 or len(bands) > 1:
                        entry[f"A_band_{key}"] = "fit-noise-limited"
                        entry[f"A_anti_band_{key}"] = "fit-noise-limited"
                # K2b per degree (review round 1 B3): unridged cond or a
                # load-bearing ridge marks the degree fit-limited.
                cf = fz[f"cond_{deg}"]
                if cf["cond_unridged"] > 1e9 or cf["ridge_loadbearing"]:
                    entry[f"A_band_{key}"] = "fit-limited"
                    entry[f"A_anti_band_{key}"] = "fit-limited"
                # A disqualified band word also withdraws the ceiling
                # label (review round 2: it keys the substitutes sentence).
                if a and entry[f"A_band_{key}"] in (
                        "unresolved-at-floor", "fit-noise-limited",
                        "fit-limited"):
                    was = ("; was VIOLATION-diagnostic"
                           if entry.get(f"ceiling_{key}")
                           == "VIOLATION-diagnostic" else "")
                    entry[f"ceiling_{key}"] = \
                        f"not-evaluated ({entry[f'A_band_{key}']}{was})"
            # Literature-convention robustness line (per-ensemble
            # whole-sample fit), disclosure only.
            for deg, col in (("deg1", "ws1"), ("deg2", "ws2")):
                aw = pooled_ratio(series("zero_sum", obs, col),
                                  series("iid", obs, col))
                entry[f"A_ws_{obs}_{deg}"] = _fmt(aw)
        report_cells[ci] = entry

    bad = [k for k, v in gates.items() if not v]
    report = {"validator_failures": fail, "gate_failures": bad,
              "gates_passed": len(gates) - len(bad),
              "gates_total": len(gates), "cells": report_cells,
              "theory_line": "ceiling test: A consistent with the degree "
                             "law iff its 2-sigma CI intersects (1, "
                             f"K/(K-1)={CEILING:.4f}]; the diagnostic "
                             "abort tests at 4 sigma. Registered "
                             "antithetic co-prediction (P-R5): A_anti on "
                             "near-Gaussian cells in (1, 2], worst 2.0 "
                             "at Hermite order m=2. Applies to ZV-1 and "
                             "ZV-2diag columns; NOT to A_ws (jointly "
                             "K-dependent fit, outside the ceiling's "
                             "hypotheses)."}
    out = Path(args.out).with_suffix(".emitter.json")
    out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in
                      ("validator_failures", "gate_failures",
                       "gates_passed", "gates_total")}, indent=1))
    if fail or bad:
        print("EMITTER: ABORT — validator or gate failure; "
              "no table is licensed.")
        sys.exit(1)
    print(f"EMITTER: PASS — full report at {out}")

    def r3(fmt):
        return f"{fmt['ratio']:.3g}" if fmt else "NA"
    for ci, e in report_cells.items():
        print(f"{e['name']:>13}: K1={r3(e['k1_plain_ratio'])} "
              f"A(o1,zv1)={r3(e['A_o1_deg1'])} [{e['A_band_o1_deg1']}] "
              f"A(o2,zv1)={r3(e['A_o2_deg1'])} [{e['A_band_o2_deg1']}] "
              f"anti(o1,zv1)={r3(e['A_anti_o1_deg1'])} "
              f"[{e['A_anti_band_o1_deg1']}]")


def _fmt(pr):
    if pr is None:
        return None
    return {"ratio": round(pr["ratio"], 6),
            "ci": [round(pr["lo"], 6), round(pr["hi"], 6)],
            "abs_log10": [round(math.log10(max(pr["abs_m"], 1e-300)), 3),
                          round(math.log10(max(pr["abs_iid"], 1e-300)), 3)],
            "g2": [round(pr["g2_m"], 2), round(pr["g2_iid"], 2)],
            "per_rep": [round(v, 6) for v in pr["per_rep"]], "n": pr["n"]}


# ------------------------------------------------------------- selftest

def selftest(device):
    ok = True
    f64 = torch.float64

    # --- gradient FD checks on synthetic in-memory data (no files)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    n, draw = 160, 5
    x_np = torch.randn(n, draw, generator=gen).numpy()
    xt = torch.as_tensor(add_intercept(x_np), device=device)
    yt = (torch.rand(n, generator=gen) < 0.4).float().to(device)
    d = xt.shape[1]

    def energy_blr(th, lam, df, s2):
        z = xt.double() @ th
        nll = torch.mean(torch.nn.functional.softplus(z) - yt.double() * z)
        thm = th[1:]
        if df is None:
            return nll + 0.5 * lam * (thm * thm).sum()
        return nll + 0.5 * (df + 1.0) * torch.log1p(thm * thm
                                                    / (df * s2)).sum()

    for lam, df in ((0.1, None), (0.1, 3.0)):
        s2 = None if df is None else (df - 2.0) / (df * lam)
        for trial in range(6):
            th = 0.4 * torch.randn(d, dtype=f64, device=device,
                                   generator=None)
            g = blr_grad(th[None, :].float(), xt, yt, lam, df, s2,
                         1024)[0].double()
            for j in (0, 1, d - 1):
                e = 1e-6
                tp, tm = th.clone(), th.clone()
                tp[j] += e
                tm[j] -= e
                fd = float((energy_blr(tp, lam, df, s2)
                            - energy_blr(tm, lam, df, s2)) / (2 * e))
                if abs(fd - float(g[j])) / max(1.0, abs(fd)) > 1e-3:
                    print(f"SELFTEST FAIL blr grad df={df} j={j}")
                    ok = False

    ngr = 4
    gmat = torch.zeros(n, ngr, device=device)
    gmat[torch.arange(n), torch.randint(0, ngr, (n,), generator=gen)] = 1.0

    def energy_hier(th, lam_beta):
        beta, u, rho = th[:d], th[d:d + ngr], th[d + ngr]
        z = xt.double() @ beta + gmat.double() @ u
        nll = torch.mean(torch.nn.functional.softplus(z) - yt.double() * z)
        return (nll + 0.5 * lam_beta * (beta[1:] ** 2).sum()
                + 0.5 * (u * u).sum() * torch.exp(-2 * rho)
                + ngr * rho + 0.5 * rho * rho)

    for trial in range(4):
        th = 0.3 * torch.randn(d + ngr + 1, dtype=f64, device=device)
        g = hier_grad(th[None, :].float(), xt, yt, gmat, 0.1,
                      1024)[0].double()
        for j in (0, d, d + ngr):
            e = 1e-6
            tp, tm = th.clone(), th.clone()
            tp[j] += e
            tm[j] -= e
            fd = float((energy_hier(tp, 0.1) - energy_hier(tm, 0.1))
                       / (2 * e))
            if abs(fd - float(g[j])) / max(1.0, abs(fd)) > 1e-3:
                print(f"SELFTEST FAIL hier grad j={j}")
                ok = False

    # --- float64 run_arm smoke on every grad family (review round 1 B1:
    # the robustness rows must not crash on dtype-mismatched matmuls)
    class SynthTarget(Target):
        def __init__(self, model):
            self.cell = {"model": model, "eta": 0.05, "name": "synth",
                         "lam": 0.1,
                         "df": 3.0 if model == "tprior" else None}
            self.chunk = 1024
            self._cache = {}
            if model == "hier":
                self._base = (xt, yt, gmat)
                self.db, self.ng = d, ngr
                self.dim = d + ngr + 1
                self.args = (0.1,)
            else:
                self._base = (xt, yt, xt)
                self.dim = d
                df = self.cell["df"]
                s2 = None if df is None else (df - 2.0) / (df * 0.1)
                self.args = (0.1, df, s2)

    for model in ("blr", "tprior", "hier"):
        st = SynthTarget(model)
        for dt in (torch.float32, torch.float64):
            try:
                rr = run_arm(st, "zero_sum", 4242, 4, 24, 12, device, dt,
                             want_hash=False)
                if rr["nonfinite"] or \
                        not bool(torch.isfinite(rr["ws1"]).all()):
                    print(f"SELFTEST FAIL {model} {dt} nonfinite")
                    ok = False
            except Exception as exc:
                print(f"SELFTEST FAIL {model} {dt} run_arm raised: {exc}")
                ok = False

    # --- coupling checks (clone of the multimodal selftest)
    gend = torch.Generator(device=device)
    gend.manual_seed(2)
    z = make_noise((40000, 8, 1), "zero_sum", gend, device)
    if float(z.sum(dim=1).abs().max()) > 1e-3:
        print("SELFTEST FAIL zero_sum row sums")
        ok = False
    if abs(float(z.var()) - 1.0) > 0.02:
        print("SELFTEST FAIL zero_sum marginal var")
        ok = False
    z0, z1 = z[:, 0, 0], z[:, 1, 0]
    corr = float(((z0 - z0.mean()) * (z1 - z1.mean())).mean()
                 / (z0.std() * z1.std()))
    if abs(corr - (-1.0 / 7.0)) > 0.02:
        print(f"SELFTEST FAIL C_star cross-corr {corr:.4f}")
        ok = False
    za = make_noise((512, 8, 4), "antithetic", gend, device)
    if float((za[:, :4] + za[:, 4:8]).abs().max()) > 1e-6:
        print("SELFTEST FAIL antithetic pairing")
        ok = False

    # --- ZV plumbing on an internal standard-Gaussian toy (grad U = x).
    # Plumbing check only; NOT a campaign cell (precommit bars synthetic
    # Gaussians from the registered table precisely because ZV-1 is exact
    # here — which is what makes it the right end-to-end check).
    class Toy:
        cell = {"model": "toy", "eta": 0.1, "steps": 400, "burnin": 200,
                "name": "toy", "reps": 64, "lam": None, "df": None}
        dim = 4

        def grad(self, x):
            return x

        def o2(self, x):
            return sigmoid(x.sum(dim=-1))
    toy = Toy()
    r = run_arm(toy, "iid", 12345, 64, 400, 200, device, torch.float32,
                want_hash=False)
    mx, mz = r["mean_x"], r["mean_z"]
    # analytic deg-1 coefficients for grad U = x: f=x_j on z=-x -> b=-I
    zv = mx - mz @ (-torch.eye(4, dtype=f64, device=device))
    v_plain = float(((mx - mx.mean(0)) ** 2).sum(1).mean())
    v_zv = float(((zv - zv.mean(0)) ** 2).sum(1).mean())
    if not v_zv < v_plain / 50:
        print(f"SELFTEST FAIL ZV plumbing: plain {v_plain:.3g} "
              f"zv {v_zv:.3g}")
        ok = False

    # --- full fit -> save -> load -> apply round-trip (review round 1 B4:
    # an orientation bug in the file round-trip would be favorable-silent;
    # the analytic-coefficient check above cannot catch it)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        fit_cell(0, toy, device, torch.float32, td,
                 {"note": "selftest round-trip"}, quick=False)
        fro = load_frozen(td, toy.cell, device, 0)
        eye_err = float((fro["deg1"][:, :4]
                         + torch.eye(4, dtype=f64, device=device))
                        .abs().max())
        if eye_err > 0.05:
            print(f"SELFTEST FAIL round-trip deg1 coef vs -I: {eye_err}")
            ok = False
        rr = run_arm(toy, "iid", 777, 32, 200, 100, device,
                     torch.float32, want_hash=False)
        row = {"G": 32, "dim": 4,
               "mean_x": rnd(rr["mean_x"]), "mean_z": rnd(rr["mean_z"]),
               "mean_w": rnd(rr["mean_w"]), "mean_o2": rnd(rr["mean_o2"]),
               "ws1_o1": rnd(rr["ws1"][:, :4]),
               "ws1_o2": rnd(rr["ws1"][:, 4]),
               "ws2_o1": rnd(rr["ws2"][:, :4]),
               "ws2_o2": rnd(rr["ws2"][:, 4])}
        cc = columns_from_row(row, fro, device)
        vp = float(((cc[("o1", "plain")]
                     - cc[("o1", "plain")].mean(0)) ** 2).sum(1).mean())
        vz = float(((cc[("o1", "zv1")]
                     - cc[("o1", "zv1")].mean(0)) ** 2).sum(1).mean())
        if not vz < vp / 50:
            print(f"SELFTEST FAIL round-trip apply: plain {vp:.3g} "
                  f"zv1 {vz:.3g}")
            ok = False

    # --- determinism
    r1 = run_arm(toy, "zero_sum", 999, 8, 60, 30, device, torch.float32)
    r2 = run_arm(toy, "zero_sum", 999, 8, 60, 30, device, torch.float32)
    if r1["state_hash_50"] != r2["state_hash_50"]:
        print("SELFTEST FAIL determinism")
        ok = False

    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--fit", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--campaign", action="store_true")
    p.add_argument("--emit", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--data-dir", default="data/real")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--frozen-dir", default="outputs")
    p.add_argument("--precommit",
                   default="campaigns/zv_composition_precommit.md")
    p.add_argument("--out", default="outputs/zv_composition.jsonl")
    p.add_argument("--cells", type=int, nargs="+", default=None,
                   help="campaign/fit: run this subset of cell indices "
                        "(GPU sharding on one node; seeds are per-(cell,"
                        "method,replicate) so partitioning changes no "
                        "sequence; concatenate shard files before --emit)")
    p.add_argument("--precommit-sha", default=None,
                   help="emit only: the LOCKED precommit blob sha256; "
                        "rows are validated against this instead of the "
                        "living file, which dated addenda may have moved")
    p.add_argument("--chunk-rows", type=int, default=512)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    if args.selftest:
        sys.exit(0 if selftest(device) else 1)

    if args.emit:
        emit(args, device)
        return

    targets = None

    def get_targets():
        return {c["name"]: Target(c, args.data_dir, args.raw_dir, device,
                                  args.chunk_rows) for c in CELLS}

    if args.fit:
        prov = provenance(args.precommit)
        Path(args.frozen_dir).mkdir(parents=True, exist_ok=True)
        targets = get_targets()
        for ci, c in enumerate(CELLS):
            if args.cells is not None and ci not in args.cells:
                continue        # per-cell resume after a partial failure
            fit_cell(ci, targets[c["name"]], device, torch.float32,
                     args.frozen_dir, prov, quick=args.quick)
        return

    if args.pilot:
        # EXPLORATORY: stability + magnitudes only; no scored metric.
        # Separate default file so pilot rows never touch the campaign
        # JSONL (campaign refuses to append to an existing file).
        prov = provenance(args.precommit)
        targets = get_targets()
        out = Path(args.out if args.out != p.get_default("out")
                   else "outputs/zv_pilot.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            sys.exit(f"refusing to append to existing {out}")
        with out.open("a") as fh:
            for ci, c in enumerate(CELLS):
                t = targets[c["name"]]
                for m in METHODS:
                    r = run_arm(t, m, PILOT_SEED_BASE + 10 * ci
                                + METHODS.index(m), 16, c["steps"],
                                c["burnin"], device, torch.float32)
                    row = {"status": "exploratory-pilot",
                           "cell_name": c["name"], "method": m,
                           "G": 16, "nonfinite": r["nonfinite"],
                           "max_sq_norm": r["max_sq_norm"],
                           "steps_per_sec": r["steps_per_sec"],
                           "wall_sec": r["wall_sec"], **prov}
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    print(f"pilot {c['name']:>13} {m:>10}: "
                          f"{r['steps_per_sec']:.0f} st/s "
                          f"maxsq={r['max_sq_norm']:.3g} "
                          f"nonfinite={r['nonfinite']}")
        return

    if args.campaign:
        if not Path(args.precommit).exists():
            sys.exit(f"refusing: no precommit at {args.precommit}")
        if not selftest(device):
            sys.exit("selftest failed — whole-campaign abort")
        targets = get_targets()
        frozen = {}
        for fci, c in enumerate(CELLS):
            path = Path(args.frozen_dir) / f"zv_frozen_{c['name']}.json"
            if not path.exists():
                sys.exit(f"refusing: no frozen coefficients at {path} "
                         f"(run --fit first)")
            frozen[c["name"]] = load_frozen(args.frozen_dir, c, device,
                                            fci)
            if frozen[c["name"]]["dim"] != targets[c["name"]].dim:
                sys.exit(f"frozen dim {frozen[c['name']]['dim']} != "
                         f"target dim {targets[c['name']].dim} for "
                         f"{c['name']}")
        # quick smokes write to their own file so they can never bypass
        # the campaign overwrite guard or pollute the accepted JSONL.
        out = Path("outputs/zv_quick.jsonl"
                   if args.quick and args.out == p.get_default("out")
                   else args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            sys.exit(f"refusing to append to existing {out} "
                     f"(applies to --quick too; remove it first)")
        prov = provenance(args.precommit)
        runs = [(ci, m, rep, torch.float32) for ci in range(len(CELLS))
                for m in METHODS for rep in REPS]
        runs += [(ci, m, 0, torch.float64) for ci in range(len(CELLS))
                 for m in METHODS]
        if args.quick:
            runs = [(0, m, 0, torch.float32) for m in METHODS]
        if args.cells is not None:
            runs = [r for r in runs if r[0] in args.cells]
        with out.open("a") as fh:
            for ci, m, rep, dtype in runs:
                c = CELLS[ci]
                row = campaign_row(ci, targets[c["name"]], m, rep, device,
                                   dtype, frozen[c["name"]], prov)
                if args.quick:
                    row["status"] = "quick-smoke"
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                print(f"c{ci} {c['name']:>13} {m:>10} r{rep} "
                      f"{row['dtype_dynamics']}: "
                      f"{row['steps_per_sec']:.0f} st/s "
                      f"maxsq={row['max_sq_norm']:.3g} "
                      f"nonfinite={row['nonfinite']}")
        print(f"wrote {out}")
        return

    print("pick a mode: --selftest / --fit / --pilot / --campaign / --emit")
    sys.exit(2)


if __name__ == "__main__":
    main()
