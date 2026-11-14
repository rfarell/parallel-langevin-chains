"""Closed-loop ChEES shared-length adaptation on coupled chains (anchor).

Implements the pre-registered protocol campaigns/chees_closedloop_precommit.md
IN ITS ENTIRETY (original draft + amendments v2/v3/v4/v5 + at-lock
rulings; the V2-0/V3-0 supersession maps decide conflicts), at
precommit sha256
4803faad6846be4ccb3d8b25868bdf7c164f75acaa8f07041e71871c560fa0b1
(v5 relocates the S-2 negative-direction probe from T=9 to T=6 on the
mathematical grounds recorded there).

Design: per-ensemble ChEES tuner (Adam on log T, lr 0.025, beta1 0,
beta2 0.95; shared-per-ensemble dual averaging on log eps targeting
harmonic-mean acceptance 0.651), rotated-Halton jitter with per-phase
rotation streams and continuous 1-based indexing, freeze at step 450 to
(Tema, exp(logepsbar)), equilibration 451-500, scored 501-650.  Arms
A/B/C = closed loop under iid / zero_sum / antithetic refresh; 2x2
fresh-score replay factorial {A-,B-learned} x {iid, zero_sum} at 300
steps with shared x0 and matched jitter; 1300-step auxiliary iid
reference at the arm-A median frozen schedule; warmup-only cross-fitted
centering shadow; trigger-only G=4 secondary variant.  V2-7 seed plan
(370M block, S-10 uniqueness); V3-6 triple-hash provenance on every
row/sidecar/selftest log; V2-10 --qc-only blinded mode; V3-5+V4-B
reading ladder.  Substrate: scripts/nuts_real_logistic.py (refactored
jitter_propose/jitter_accept; frozen legacy oracle pinned at
scripts/legacy/nuts_real_logistic_36d66d6a.py for S-7).
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nuts_real_logistic as nrl  # noqa: E402  (refactored substrate)

# --------------------------------------------------------------------------
# Registered constants (V2-8 / V3-4)
# --------------------------------------------------------------------------
ADAM_LR = 0.025
ADAM_BETA1 = 0.0
ADAM_BETA2 = 0.95
ADAM_STAB = 1e-8
DA_GAMMA = 0.05
DA_T0 = 10.0
DA_KAPPA = 0.75
DA_TARGET = 0.651
ALPHA_FLOOR = 1e-10
L_MAX = 256
T_BOX_LO = 0.5           # times eps
T_BOX_HI = 250.0         # times eps
EPS_BOX_LO = 1.0 / 8.0   # times eps_0
EPS_BOX_HI = 8.0         # times eps_0
N_WARM = 450
N_EQUIL = 50
N_SCORE = 150
N_TOTAL = N_WARM + N_EQUIL + N_SCORE          # 650
REPLAY_BURN = 150
REPLAY_SCORE = 150
REPLAY_TOTAL = REPLAY_BURN + REPLAY_SCORE     # 300
REF_STEPS = 1300
REF_BURN = 650
REF_REPS = 16
REF_CHAINS = 16
EMA_RATE = 0.1
DIVERGENCE = 1000.0      # campaign constant (nuts_real_logistic)
K_CHAINS = 8
G4_GROUP = 4

DATASETS = ["wdbc", "spambase", "adult"]
PRIORS = [0.1, 0.3]
DS_IDX = {"wdbc": 0, "spambase": 1, "adult": 2}
PRIOR_IDX = {0.1: 0, 0.3: 1}
CELL_REPS = {"wdbc": 128, "spambase": 96, "adult": 48}   # DATASET_DEFAULTS
ARMS = [("A", "iid"), ("B", "zero_sum"), ("C", "antithetic")]
REPLAY_RUNS = [("A", "iid"), ("A", "zero_sum"), ("B", "iid"), ("B", "zero_sum")]

# V2-7 seed plan: base + named streams (18 per design; +530,000 reserved).
SEED_BASE0 = 370_000_000
SEED_RANGE = (370_000_000, 397_999_999)
STREAMS = {
    "probe": 1_729,
    "arm_A": 100_000,
    "arm_B": 110_000,
    "arm_C": 120_000,
    "replay_A_iid": 200_000,
    "replay_A_zero_sum": 210_000,
    "replay_B_iid": 220_000,
    "replay_B_zero_sum": 230_000,
    "replay_x0": 240_000,
    "reference": 300_000,
    "rot_closed": 400_000,
    "rot_replay": 410_000,
    "rot_reference": 420_000,
    "rot_g4": 430_000,
    "g4_A": 500_000,
    "g4_B": 510_000,
    "g4_C": 520_000,
    "g4_reserved": 530_000,
}
SELFTEST_SEEDS = [1_250_001, 1_250_002,
                  1_250_010, 1_250_011, 1_250_012, 1_250_013, 1_250_014, 1_250_015,
                  1_250_020, 1_250_021, 1_250_030, 1_250_031, 1_250_040, 1_250_050,
                  1_250_060, 1_250_070, 1_250_080, 1_250_081]
ARCHIVED_READONLY = [320_050_113, 320_050_214, 320_050_778]
S10_EXPECTED_TOTAL = 345

LEGACY_PIN = "36d66d6a77dbe2336a783cbf89307ebef4301030ad22f0dc9ccddb89671113ec"
LEGACY_PATH = Path(__file__).resolve().parent / "legacy" / "nuts_real_logistic_36d66d6a.py"
GOLD_PATH = Path("outputs/sweeps/rr_extrapolation_gold.jsonl")
GOLD_PIN = "b376b814f282cdddadf0890031306f8fa376fafeb0405908b31d11f314cc55fe"
PRECOMMIT_PATH = Path("campaigns/chees_closedloop_precommit.md")
# S-7 archived spambase/0.3 replicate-0 jitter_shared configuration (V2-5/V3-3)
S7_L_SEED = 320_050_778
S7_DYN = {"iid": 320_050_113, "zero_sum": 320_050_214}
S7_EPS_ARCHIVED = 1.6
S7_TRACEVAR_ARCHIVED = {"iid": 0.43009480834007263, "zero_sum": 0.3671760559082031}
S7_FIX_FULL_SHA = "ed994c17298653a81d0a2972e582d2ada3322e71853c5975cef705d59588b557"
S7_FIX_SUB_SHA = "4148ee7db973abdd14f3af1c007e6841193ab8fd23f26d94640087f6e4194be9"
S7_FIX_GAUSS_SHA = "3044a69c45562ec641d48722da575683c79ea7dd2f20274fe07c4569e9894248"

SELFTEST_LAMS = np.logspace(np.log10(0.25), np.log10(4.0), 8)
SELFTEST_EPS0 = 0.25
IS_STUB = bool(getattr(cp, "IS_NUMPY_STUB", False))
LOCK_MANIFEST_DEFAULT = Path("campaigns/chees_executable_lock.json")
S7_AUDIT_ARTIFACTS = ["chees_s7_gaussian_gpu.npz", "chees_s7_spambase_iid_gpu.npz",
                      "chees_s7_spambase_zero_sum_gpu.npz"]
# Registered S-7 per-step p0 fixture geometry (fix-verification defect 2)
S7_P0_SHAPES = {"chees_s7_gaussian_gpu.npz": (200, 512, 8),
                "chees_s7_spambase_iid_gpu.npz": (300, 768, 58),
                "chees_s7_spambase_zero_sum_gpu.npz": (300, 768, 58)}
HASH_KEYS = ("chees_script_sha256", "nuts_substrate_sha256", "precommit_sha256")
PRODUCTION_BACKEND = "cupy"
TRAJ_STEPS_REGISTERED = list(range(10, N_WARM + 1, 10))
TRAJ_KEYS = ("step", "T_med", "T_q25", "T_q75", "eps_med", "eps_q25", "eps_q75",
             "dghat_med", "dghat_q25", "dghat_q75")


def full_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triple_hash():
    """V3-6 triple-hash provenance, full 64-hex (overrides 16-hex convention)."""
    return {
        "chees_script_sha256": full_sha256(__file__),
        "nuts_substrate_sha256": full_sha256(Path(__file__).resolve().parent
                                             / "nuts_real_logistic.py"),
        "precommit_sha256": full_sha256(PRECOMMIT_PATH),
    }


def provenance():
    import scipy
    return {
        "argv": " ".join(sys.argv),
        **triple_hash(),
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__,
        "scipy_version": scipy.__version__,
        "cupy_backend": "numpy-stub" if IS_STUB else "cupy",
        "git_hash": nrl.git_hash(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }


def write_lock_manifest(path):
    """Create the EXTERNAL executable-lock manifest (audit finding 4) -- run
    exactly once AT the executable-hash-lock step (V2-10 step 7).  The
    validator thereafter compares every artifact against this immutable file,
    never against the currently executing bytes."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing lock manifest {path}")
    manifest = {**triple_hash(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": ("ChEES executable hash lock (V2-10 step 7 / V3-6 / "
                         "code-audit finding 4). Immutable; any change after "
                         "creation is governed by the V2-9 defect procedure.")}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(f"lock manifest written: {path}")
    for key in HASH_KEYS:
        print(f"  {key} = {manifest[key]}")


def load_lock_manifest(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"LOCK MANIFEST MISSING: {path} -- the validator refuses to run "
            f"without the immutable executable-hash-lock manifest (audit "
            f"finding 4; create it at the hash-lock step with "
            f"--write-lock-manifest)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in HASH_KEYS:
        assert key in manifest and len(manifest[key]) == 64, \
            f"lock manifest {path} missing/short {key}"
    return manifest


def seed_base(dataset, prior, replicate):
    return (SEED_BASE0 + DS_IDX[dataset] * 10_000_000
            + PRIOR_IDX[prior] * 5_000_000 + replicate * 1_000_000)


def stream_seed(dataset, prior, replicate, name):
    return seed_base(dataset, prior, replicate) + STREAMS[name]


def enumerate_campaign_seeds():
    vals = []
    for ds in DATASETS:
        for prior in PRIORS:
            for rep in range(3):
                for name in STREAMS:
                    vals.append(stream_seed(ds, prior, rep, name))
    return vals


def s10_seed_uniqueness(verbose=True):
    """S-10 (V2-7/V3-4): 324 campaign stream values in range, plus 18 selftest
    seeds and 3 archived read-only replays; global uniqueness; total 345."""
    campaign = enumerate_campaign_seeds()
    assert len(campaign) == 324, f"expected 324 campaign stream values, got {len(campaign)}"
    for v in campaign:
        assert SEED_RANGE[0] <= v <= SEED_RANGE[1], f"campaign seed {v} outside {SEED_RANGE}"
    full = campaign + SELFTEST_SEEDS + ARCHIVED_READONLY
    assert len(full) == S10_EXPECTED_TOTAL, f"expected {S10_EXPECTED_TOTAL}, got {len(full)}"
    assert len(set(full)) == len(full), "seed collision detected"
    if verbose:
        print(f"S-10 seed uniqueness: {len(campaign)} campaign stream values in "
              f"[{SEED_RANGE[0]},{SEED_RANGE[1]}], +{len(SELFTEST_SEEDS)} selftest "
              f"+{len(ARCHIVED_READONLY)} archived read-only; total {len(full)} "
              f"all unique -> PASS")
    return len(full)


# --------------------------------------------------------------------------
# Rotated-Halton jitter (V2-8 step 1, V2-J)
# --------------------------------------------------------------------------

def halton2(n):
    """Base-2 radical inverse (van der Corput) of 1-based index n, float64."""
    h, f = 0.0, 0.5
    while n > 0:
        h += f * (n & 1)
        n >>= 1
        f *= 0.5
    return h


def draw_rotations(seed, count):
    """Cranley-Patterson rotation offsets u ~ U(0,1), float64, from one numpy
    default_rng(seed) (V2-7 rule).  Returns np.ndarray (count,)."""
    return np.random.default_rng(seed).random(count)


def draw_rotations_g4(seed, n_groups, reps):
    """+430,000 stream draw order (V3-5): first the n_groups group rotations in
    group order, then the reps per-ensemble fork rotations in ensemble order."""
    rng = np.random.default_rng(seed)
    return rng.random(n_groups), rng.random(reps)


# --------------------------------------------------------------------------
# ChEES gradient (section-2 step 5 / V2-2 / V3-4)
# --------------------------------------------------------------------------

def chees_gradient(x_cur, x_prop, p_prop, alpha, t, centers=None):
    """Acceptance-weighted, ensemble-centered ChEES gradient estimate.

    x_cur/x_prop/p_prop: (units, C, dim) float32 (current states, proposed
    states, trajectory-endpoint momenta); alpha: (units, C) float64 totalized
    weights; t: (units,) float64 jittered trajectory lengths.
    centers: None -> per-unit sample means of current and proposed states
    (stock Algorithm 1, the sole primary estimand); else a pair
    (c_cur, c_prop) of (units, 1, dim) float64 arrays (cross-fitted or exact
    centers; V2-2).  Returns (ghat (units,) float64, n_all_zero int).
    """
    xc = x_cur.astype(cp.float64)
    xp = x_prop.astype(cp.float64)
    pp = p_prop.astype(cp.float64)
    if centers is None:
        c_cur = xc.mean(axis=1, keepdims=True)
        c_prop = xp.mean(axis=1, keepdims=True)
    else:
        c_cur, c_prop = centers
    dc = xc - c_cur
    dp = xp - c_prop
    dsq = (dp * dp).sum(axis=2) - (dc * dc).sum(axis=2)
    inner = (dp * pp).sum(axis=2)
    g_k = t[:, None] * dsq * inner
    # Audit finding 2: a rejected nonfinite proposal (alpha = 0, g_k possibly
    # NaN/inf) must contribute EXACTLY nothing -- mask zero-weight terms
    # BEFORE the multiplication; divide by the EXACT weight sum whenever it
    # is nonzero (the 1e-10 floor belongs to the harmonic-acceptance
    # statistic only, never the ChEES denominator); ghat = 0 only when the
    # weight sum is exactly zero.
    g_masked = cp.where((alpha > 0.0) & cp.isfinite(g_k), g_k, 0.0)
    wsum = alpha.sum(axis=1)
    zero = wsum == 0.0
    denom = cp.where(zero, 1.0, wsum)
    ghat = cp.where(zero, 0.0, (alpha * g_masked).sum(axis=1) / denom)
    return ghat, int(zero.sum())


def selftest_gradient_totality():
    """Audit finding 2 unit tests: (A) a nonfinite zero-alpha proposal
    contributes exactly nothing (others' weighted mean preserved); (A2) a
    nonfinite proposal POSITION (center contamination) still yields a finite
    gradient (0 by the finiteness mask); (B) positive weights summing below
    1e-10 divide by the EXACT sum, never the harmonic floor; (C) all-zero
    weights give ghat = 0 and increment the counter."""
    t = cp.asarray([1.0], dtype=cp.float64)
    x_cur = cp.zeros((1, 3, 1), dtype=cp.float32)
    p_fin = cp.asarray([[[2.0], [4.0], [6.0]]], dtype=cp.float32)
    x_fin = cp.asarray([[[1.0], [3.0], [5.0]]], dtype=cp.float32)
    # (A) nonfinite endpoint momentum on a zero-alpha chain
    p_bad = cp.asarray([[[2.0], [4.0], [np.inf]]], dtype=cp.float32)
    alpha = cp.asarray([[0.5, 0.25, 0.0]], dtype=cp.float64)
    ghat, nz = chees_gradient(x_cur, x_fin, p_bad, alpha, t)
    # manual: c_prop=3 -> dp=(-2,0,2); dsq=(4,0,4); inner=(dp*p)=(-4,0,+inf
    # masked); g_k=(dsq*inner)=(-16,0,masked)
    expected = (0.5 * (4.0 * (-2.0 * 2.0)) + 0.25 * 0.0) / 0.75
    got = float(ghat.get()[0])
    assert np.isfinite(got) and abs(got - expected) < 1e-12, \
        f"Sgrad(A): got {got}, expected {expected}"
    assert nz == 0
    # (A2) nonfinite proposal POSITION: centers contaminated -> masked to 0
    x_bad = cp.asarray([[[1.0], [3.0], [np.inf]]], dtype=cp.float32)
    ghat, nz = chees_gradient(x_cur, x_bad, p_fin, alpha, t)
    got = float(ghat.get()[0])
    assert got == 0.0 and np.isfinite(got), f"Sgrad(A2): got {got}"
    # (B) positive weights below the harmonic floor: exact-denominator rule
    x2 = cp.asarray([[[1.0], [3.0]]], dtype=cp.float32)
    p2 = cp.asarray([[[2.0], [4.0]]], dtype=cp.float32)
    a2 = cp.asarray([[1e-12, 2e-12]], dtype=cp.float64)
    ghat, _ = chees_gradient(cp.zeros((1, 2, 1), dtype=cp.float32), x2, p2, a2, t)
    # c_prop=2 -> dp=(-1,1); dsq=(1,1); inner=(-2,4); g=(-2,4)
    expected = (1e-12 * -2.0 + 2e-12 * 4.0) / 3e-12
    got = float(ghat.get()[0])
    assert abs(got - expected) < 1e-9, f"Sgrad(B): got {got}, expected {expected}"
    assert abs(got - 2.0) < 1e-9, "Sgrad(B): exact-sum division violated"
    # (C) all-zero weights: ghat = 0, counter increments
    a0 = cp.asarray([[0.0, 0.0, 0.0]], dtype=cp.float64)
    ghat, nz = chees_gradient(x_cur, x_fin, p_fin, a0, t)
    assert float(ghat.get()[0]) == 0.0 and nz == 1
    print("Sgrad PASS: nonfinite zero-alpha proposal masked exactly; "
          "center-contamination total (ghat=0); exact-denominator below 1e-10; "
          "all-zero-weight rule")


def crossfit_centers(x_cur, x_prop, group=G4_GROUP):
    """V2-2 cross-fitted centers: for ensemble e in preassigned consecutive
    groups of `group`, the centers are the means over the OTHER group members'
    chains (both current and proposed, per the at-lock ruling).  Inputs
    (reps, K, dim) float32; returns ((reps,1,dim), (reps,1,dim)) float64."""
    reps, k_n, dim = x_cur.shape
    n_groups = reps // group
    out = []
    for arr in (x_cur, x_prop):
        a = arr.astype(cp.float64).reshape(n_groups, group, k_n, dim)
        gsum = a.sum(axis=(1, 2), keepdims=True)          # (n_groups,1,1,dim)
        esum = a.sum(axis=2, keepdims=True)               # (n_groups,group,1,dim)
        others = (gsum - esum) / float((group - 1) * k_n)
        out.append(others.reshape(reps, 1, dim))
    return out[0], out[1]


class ShadowTuner:
    """Plug-in re-integration of the V2-8 Adam/EMA recurrence at fixed
    recorded dynamics (V2-2): identical init, g = shadow gradient, T clamped
    against the PRIMARY eps trajectory, frozen at step N_WARM.  Never touches
    states, jitter, eps, or acceptance."""

    def __init__(self, n_units, eps0, freeze_at=N_WARM):
        self.logT = cp.full((n_units,), math.log(eps0), dtype=cp.float64)
        self.v = cp.zeros((n_units,), dtype=cp.float64)
        self.tema = cp.zeros((n_units,), dtype=cp.float64)
        self.freeze_at = freeze_at
        self.t_star = None

    def update(self, n, ghat, eps_primary):
        self.v = ADAM_BETA2 * self.v + (1.0 - ADAM_BETA2) * ghat * ghat
        vhat = self.v / (1.0 - ADAM_BETA2 ** n)
        self.logT = self.logT + ADAM_LR * ghat / (cp.sqrt(vhat) + ADAM_STAB)
        t_raw = cp.exp(self.logT)
        t_new = cp.clip(t_raw, T_BOX_LO * eps_primary, T_BOX_HI * eps_primary)
        self.logT = cp.log(t_new)
        self.tema = (1.0 - EMA_RATE) * self.tema + EMA_RATE * t_new
        if n == self.freeze_at:
            self.t_star = self.tema.copy()


def qcounters():
    return {ph: {"t_clamp": 0, "eps_clamp": 0, "l_max": 0, "divergent": 0,
                 "all_zero_weight": 0, "proposals": 0, "ensemble_steps": 0}
            for ph in ("warmup", "equil", "scored")}


def phase_of(n, n_warm=N_WARM, n_equil=N_EQUIL):
    if n <= n_warm:
        return "warmup"
    if n <= n_warm + n_equil:
        return "equil"
    return "scored"


# --------------------------------------------------------------------------
# Closed-loop runner (V2-8 executable pseudocode, verbatim recurrences)
# --------------------------------------------------------------------------

def run_closed_loop(potential, dim, reps, chains, coupling, eps0, dyn_seed,
                    u_rot, group_size=1, u_fork=None, probe=None,
                    crossfit=True, exact_center_shadow=False,
                    n_warm=N_WARM, n_equil=N_EQUIL, n_score=N_SCORE,
                    log_traj=True, full_traj=False, capture=None):
    """One closed-loop arm.  u_rot: warmup rotations (per ensemble for the
    primary variant; per GROUP for G=4).  u_fork: post-freeze per-ensemble
    rotations (G=4 fork; None -> u_rot, the primary variant).  Halton index =
    outer step n, 1-based, continuous 1..n_total, never reset (V2-J)."""
    n_total = n_warm + n_equil + n_score
    n_units = reps // group_size
    assert reps % group_size == 0
    cp.random.seed(dyn_seed)
    x = cp.zeros((reps * chains, dim), dtype=cp.float32)   # x_0 = 0 (V2-8)
    u, g = potential(x)
    # float64 tuner state (V3-4 dtype registration)
    logT = cp.full((n_units,), math.log(eps0), dtype=cp.float64)
    t_prev = cp.exp(logT)                                  # T_0 = eps_0
    eps_prev = cp.full((n_units,), float(eps0), dtype=cp.float64)
    v_adam = cp.zeros((n_units,), dtype=cp.float64)
    hbar = cp.zeros((n_units,), dtype=cp.float64)
    logepsbar = cp.full((n_units,), math.log(eps0), dtype=cp.float64)
    tema = cp.zeros((n_units,), dtype=cp.float64)
    mu = math.log(10.0 * eps0)
    log_eps_lo, log_eps_hi = math.log(eps0 * EPS_BOX_LO), math.log(eps0 * EPS_BOX_HI)
    u_rot_cp = cp.asarray(u_rot, dtype=cp.float64)
    u_fork_cp = u_rot_cp if u_fork is None else cp.asarray(u_fork, dtype=cp.float64)
    shadow_cf = (ShadowTuner(n_units, eps0, n_warm)
                 if (crossfit and group_size == 1) else None)
    shadow_ex = ShadowTuner(n_units, eps0, n_warm) if exact_center_shadow else None
    probe_cp = cp.asarray(probe, dtype=cp.float64) if probe is not None else None

    counters = qcounters()
    ug = {ph: 0 for ph in counters}
    bp = {ph: 0 for ph in counters}
    wc = {ph: 0.0 for ph in counters}
    l_sum = {ph: 0.0 for ph in counters}
    l_n = {ph: 0 for ph in counters}
    acc_cp = {ph: cp.zeros((), dtype=cp.float64) for ph in counters}
    steps_in = {"warmup": n_warm, "equil": n_equil, "scored": n_score}
    freeze_clamp = 0
    freeze_probe = None
    warm_abar_mean = warm_abar_median = float("nan")
    scored_abar_acc = cp.zeros((), dtype=cp.float64)
    t_star_u = eps_star_u = None       # (n_units,) frozen
    t_star_ens = eps_star_ens = None   # (reps,) broadcast for post-freeze use
    ens_acc = cp.zeros((reps, dim), dtype=cp.float32)
    single_acc = cp.zeros((reps, dim), dtype=cp.float32)
    probe_acc = cp.zeros((reps,), dtype=cp.float64)
    m2c0_acc = cp.zeros((dim,), dtype=cp.float64)
    freeze_chain0_sq = None
    windows = 0
    traj = {"step": [], "T_med": [], "T_q25": [], "T_q75": [],
            "eps_med": [], "eps_q25": [], "eps_q75": [],
            "dghat_med": [], "dghat_q25": [], "dghat_q75": []}
    # audit rec (c): accumulate full trajectories ON DEVICE; one D2H transfer
    # after the loop, outside every timed region.
    ft_dev = ({k: cp.full((n_warm, n_units), cp.nan, dtype=cp.float64) for k in
               ("T", "eps", "ghat", "ghat_cf", "abar")} if full_traj else None)
    tlen_acc = cp.zeros((reps,), dtype=cp.float64)   # audit rec (b): L*eps per ens

    cp.cuda.Stream.null.synchronize()
    ph_t0 = time.perf_counter()
    for n in range(1, n_total + 1):
        ph = phase_of(n, n_warm, n_equil)
        if n <= n_warm:
            u_now, t_use, eps_use, units_now = u_rot_cp, t_prev, eps_prev, n_units
            grp_now = group_size
        else:
            # frozen schedule; per-ensemble jitter (G=4 forks to u_e)
            assert bool((t_star_ens == t_star_frozen).all()), "T* drifted post-freeze"
            assert bool((eps_star_ens == eps_star_frozen).all()), "eps* drifted post-freeze"
            u_now, t_use, eps_use, units_now = u_fork_cp, t_star_ens, eps_star_ens, reps
            grp_now = 1
        h = (halton2(n) + u_now) % 1.0
        t_len = h * t_use
        lf = cp.ceil(t_len / eps_use)                      # float64 before cast
        counters[ph]["l_max"] += int((lf > L_MAX).sum())
        l_unit = cp.minimum(cp.maximum(lf, 1.0), float(L_MAX)).astype(cp.int32)
        l_ens = cp.repeat(l_unit, grp_now) if grp_now > 1 else l_unit
        eps_ens = cp.repeat(eps_use, grp_now) if grp_now > 1 else eps_use
        l_row = cp.repeat(l_ens, chains)
        eps_w = cp.repeat(eps_ens.astype(cp.float32), chains)[:, None]  # cast once/step
        p0 = nrl.make_noise((reps, chains, dim), coupling).reshape(-1, dim)
        x_old = x
        xf, uf, gf, pf, log_alpha, alpha = nrl.jitter_propose(
            x, u, g, p0, eps_w, l_row, potential)
        (x, u, g), accept, u_unif = nrl.jitter_accept(
            x, u, g, xf, uf, gf, log_alpha, acc_cp[ph])
        counters[ph]["divergent"] += int((log_alpha < -DIVERGENCE).sum())
        counters[ph]["proposals"] += reps * chains
        counters[ph]["ensemble_steps"] += units_now
        ug[ph] += int(l_row.sum())
        bp[ph] += int(l_unit.max())
        l_sum[ph] += float(l_row.sum())
        l_n[ph] += reps * chains
        # harmonic-mean acceptance statistic (registered floor), per tuner unit
        alpha_u = alpha.reshape(n_units, group_size * chains)
        abar = 1.0 / (1.0 / cp.maximum(alpha_u, ALPHA_FLOOR)).mean(axis=1)
        if ph == "scored":
            scored_abar_acc += abar.mean()
        if n <= n_warm:
            # 5a. dual averaging on log eps (Hoffman-Gelman 2014, verbatim)
            hbar = (1.0 - 1.0 / (n + DA_T0)) * hbar + (1.0 / (n + DA_T0)) * (DA_TARGET - abar)
            log_eps_raw = mu - (math.sqrt(n) / DA_GAMMA) * hbar
            log_eps = cp.clip(log_eps_raw, log_eps_lo, log_eps_hi)
            counters[ph]["eps_clamp"] += int((log_eps != log_eps_raw).sum())
            w_n = float(n) ** (-DA_KAPPA)
            logepsbar = w_n * log_eps + (1.0 - w_n) * logepsbar   # clipped value feeds average
            eps_new = cp.exp(log_eps)
            # 5b. ChEES gradient + Adam ascent on log T (g used directly)
            x3o = x_old.reshape(n_units, group_size * chains, dim)
            xf3 = xf.reshape(n_units, group_size * chains, dim)
            pf3 = pf.reshape(n_units, group_size * chains, dim)
            ghat, nzero = chees_gradient(x3o, xf3, pf3, alpha_u, t_len)
            counters[ph]["all_zero_weight"] += nzero
            v_adam = ADAM_BETA2 * v_adam + (1.0 - ADAM_BETA2) * ghat * ghat
            vhat = v_adam / (1.0 - ADAM_BETA2 ** n)
            logT = logT + ADAM_LR * ghat / (cp.sqrt(vhat) + ADAM_STAB)
            # 5c. clamp order: T-update -> clamp against eps_n
            t_raw = cp.exp(logT)
            t_new = cp.clip(t_raw, T_BOX_LO * eps_new, T_BOX_HI * eps_new)
            counters[ph]["t_clamp"] += int((t_new != t_raw).sum())
            logT = cp.log(t_new)
            # 5d. EMA
            tema = (1.0 - EMA_RATE) * tema + EMA_RATE * t_new
            ghat_cf = None
            if shadow_cf is not None:
                cfc = crossfit_centers(x_old.reshape(reps, chains, dim),
                                       xf.reshape(reps, chains, dim))
                ghat_cf, _ = chees_gradient(x3o, xf3, pf3, alpha_u, t_len, centers=cfc)
                shadow_cf.update(n, ghat_cf, eps_new)
            if shadow_ex is not None:
                zc = cp.zeros((n_units, 1, dim), dtype=cp.float64)
                ghat_ex, _ = chees_gradient(x3o, xf3, pf3, alpha_u, t_len,
                                            centers=(zc, zc))
                shadow_ex.update(n, ghat_ex, eps_new)
            if log_traj and n % 10 == 0:
                traj["step"].append(n)
                for key, arr in (("T", t_new), ("eps", eps_new)):
                    q = cp.percentile(arr, cp.asarray([25.0, 50.0, 75.0]))
                    q = [float(z) for z in q.get()]
                    traj[f"{key}_q25"].append(q[0])
                    traj[f"{key}_med"].append(q[1])
                    traj[f"{key}_q75"].append(q[2])
                if ghat_cf is not None:
                    dq = cp.percentile(ghat_cf - ghat, cp.asarray([25.0, 50.0, 75.0]))
                    dq = [float(z) for z in dq.get()]
                else:
                    dq = [float("nan")] * 3
                traj["dghat_q25"].append(dq[0])
                traj["dghat_med"].append(dq[1])
                traj["dghat_q75"].append(dq[2])
            if full_traj:
                ft_dev["T"][n - 1] = t_new
                ft_dev["eps"][n - 1] = eps_new
                ft_dev["ghat"][n - 1] = ghat
                if ghat_cf is not None:
                    ft_dev["ghat_cf"][n - 1] = ghat_cf
                ft_dev["abar"][n - 1] = abar
            t_prev, eps_prev = t_new, eps_new
            if n == n_warm:
                # 6. FREEZE: T* = Tema, eps* = exp(logepsbar); box assert/clip
                warm_abar_mean = float(abar.mean())
                warm_abar_median = float(cp.median(abar))
                t_star_u = tema.copy()
                eps_star_u = cp.exp(logepsbar)
                lo, hi = T_BOX_LO * eps_star_u, T_BOX_HI * eps_star_u
                bad = (t_star_u < lo) | (t_star_u > hi)
                freeze_clamp = int(bad.sum())
                t_star_u = cp.clip(t_star_u, lo, hi)
                t_star_ens = (cp.repeat(t_star_u, group_size)
                              if group_size > 1 else t_star_u)
                eps_star_ens = (cp.repeat(eps_star_u, group_size)
                                if group_size > 1 else eps_star_u)
                t_star_frozen = t_star_ens.copy()
                eps_star_frozen = eps_star_ens.copy()
                x3 = x.reshape(reps, chains, dim)
                freeze_chain0_sq = (x3[:, 0, :].astype(cp.float64) ** 2).get()
                if probe_cp is not None:
                    freeze_probe = (x3[:, 0, :].astype(cp.float64)
                                    @ probe_cp).get().tolist()
        if ph == "scored":
            x3 = x.reshape(reps, chains, dim)
            ens_acc += x3.mean(axis=1)
            single_acc += x3[:, 0, :]
            m2c0_acc += (x3[:, 0, :].astype(cp.float64) ** 2).mean(axis=0)
            tlen_acc += l_ens.astype(cp.float64) * eps_ens   # realized L*eps_e
            if probe_cp is not None:
                probe_acc += x3[:, 0, :].astype(cp.float64) @ probe_cp
            windows += 1
        if capture is not None and n in capture["steps"]:
            capture["data"][n] = {
                "h": h.get().copy(), "l_row": l_row.get().copy(),
                "p0": p0.get().copy(), "u_unif": u_unif.get().copy(),
                "log_alpha": log_alpha.get().copy(),
                "x_pre": x_old.get().copy(), "x_post": x.get().copy(),
                "u_post": u.get().copy(), "g_post": g.get().copy(),
                "T_use": t_use.get().copy(), "eps_use": eps_use.get().copy(),
                "u_rot_now": u_now.get().copy()}
        if n == n_warm or n == n_warm + n_equil or n == n_total:
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            wc[ph] += t1 - ph_t0
            ph_t0 = t1
    wdiv = max(windows, 1)
    res = {
        "n_units": n_units, "group_size": group_size, "windows": windows,
        "ens": ens_acc / wdiv, "single": single_acc / wdiv,
        "m_probe": (probe_acc / wdiv).get().tolist() if probe is not None else None,
        "m_probe_freeze": freeze_probe,
        "m2_chain0_scored": (m2c0_acc / wdiv).get(),
        "freeze_chain0_sq": freeze_chain0_sq,
        "T_star": t_star_u.get(), "eps_star": eps_star_u.get(),
        "T_star_cf": (shadow_cf.t_star.get() if shadow_cf is not None else None),
        "T_star_exact_center": (shadow_ex.t_star.get() if shadow_ex is not None else None),
        "freeze_clamp": freeze_clamp,
        "warmup_end_abar_mean": warm_abar_mean,
        "warmup_end_abar_median": warm_abar_median,
        "scored_abar_mean": float(scored_abar_acc.get()) / max(n_score, 1),
        "accept": {ph: float(acc_cp[ph].get()) / max(steps_in[ph], 1)
                   for ph in counters},
        "counters": counters, "ug": ug, "bp": bp, "wc": wc,
        "mean_l": {ph: (l_sum[ph] / l_n[ph] if l_n[ph] else float("nan"))
                   for ph in counters},
        "mean_realized_tlen": (tlen_acc / wdiv).get().tolist(),
        "traj": traj,
        "full_traj": ({k: v.get() for k, v in ft_dev.items()}
                      if full_traj else None),   # single D2H, untimed (rec c)
        "frozenness_ok": True,   # in-run post-freeze asserts all passed
        "halton_index_ranges": {"warmup": [1, n_warm],
                                "equil": [n_warm + 1, n_warm + n_equil],
                                "scored": [n_warm + n_equil + 1, n_total]},
    }
    return res


# --------------------------------------------------------------------------
# Frozen-schedule runner (replay factorial, reference, S-3a scoring passes)
# --------------------------------------------------------------------------

def run_frozen(potential, dim, reps, chains, coupling, u_rot, t_sched, eps_sched,
               steps, burnin, x0=None, probe=None, snapshots=(),
               halton_start=1):
    """Exogenous frozen per-ensemble schedule (T_e, eps_e); no adaptation.
    Halton index runs halton_start..halton_start+steps-1, 1-based, no resets.
    Caller seeds cp.random BEFORE calling (so shared-x0 draws precede)."""
    u_rot_cp = cp.asarray(u_rot, dtype=cp.float64)
    t_cp = cp.asarray(t_sched, dtype=cp.float64)
    eps_cp = cp.asarray(eps_sched, dtype=cp.float64)
    if x0 is None:
        x = cp.zeros((reps * chains, dim), dtype=cp.float32)
    else:
        x = cp.asarray(x0, dtype=cp.float32).reshape(reps * chains, dim).copy()
    u, g = potential(x)
    probe_cp = cp.asarray(probe, dtype=cp.float64) if probe is not None else None
    counters = {ph: {"l_max": 0, "divergent": 0, "proposals": 0}
                for ph in ("burnin", "scored")}
    ug = {ph: 0 for ph in counters}
    bp = {ph: 0 for ph in counters}
    wc = {ph: 0.0 for ph in counters}
    l_sum = {ph: 0.0 for ph in counters}
    l_n = {ph: 0 for ph in counters}
    acc_cp = {ph: cp.zeros((), dtype=cp.float64) for ph in counters}
    ens_acc = cp.zeros((reps, dim), dtype=cp.float32)
    single_acc = cp.zeros((reps, dim), dtype=cp.float32)
    probe_acc = cp.zeros((reps,), dtype=cp.float64)
    windows = 0
    snaps = {}
    cp.cuda.Stream.null.synchronize()
    ph_t0 = time.perf_counter()
    for i in range(1, steps + 1):
        n = halton_start + i - 1
        ph = "burnin" if i <= burnin else "scored"
        h = (halton2(n) + u_rot_cp) % 1.0
        t_len = h * t_cp
        lf = cp.ceil(t_len / eps_cp)
        counters[ph]["l_max"] += int((lf > L_MAX).sum())
        l_ens = cp.minimum(cp.maximum(lf, 1.0), float(L_MAX)).astype(cp.int32)
        l_row = cp.repeat(l_ens, chains)
        eps_w = cp.repeat(eps_cp.astype(cp.float32), chains)[:, None]
        p0 = nrl.make_noise((reps, chains, dim), coupling).reshape(-1, dim)
        xf, uf, gf, pf, log_alpha, alpha = nrl.jitter_propose(
            x, u, g, p0, eps_w, l_row, potential)
        (x, u, g), accept, u_unif = nrl.jitter_accept(
            x, u, g, xf, uf, gf, log_alpha, acc_cp[ph])
        counters[ph]["divergent"] += int((log_alpha < -DIVERGENCE).sum())
        counters[ph]["proposals"] += reps * chains
        ug[ph] += int(l_row.sum())
        bp[ph] += int(l_ens.max())
        l_sum[ph] += float(l_row.sum())
        l_n[ph] += reps * chains
        if i > burnin:
            x3 = x.reshape(reps, chains, dim)
            ens_acc += x3.mean(axis=1)
            single_acc += x3[:, 0, :]
            if probe_cp is not None:
                probe_acc += x3[:, 0, :].astype(cp.float64) @ probe_cp
            windows += 1
        if i in snapshots:
            snaps[i] = x.reshape(reps, chains, dim).get().copy()
        if i == burnin or i == steps:
            cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            wc[ph] += t1 - ph_t0
            ph_t0 = t1
    return {
        "windows": windows,
        "ens": (ens_acc / windows if windows else ens_acc),
        "single": (single_acc / windows if windows else single_acc),
        "m_probe": ((probe_acc / windows).get().tolist()
                    if (probe is not None and windows) else None),
        "accept": {ph: (float(acc_cp[ph].get()) / max(1, (burnin if ph == "burnin"
                        else steps - burnin))) for ph in counters},
        "counters": counters, "ug": ug, "bp": bp, "wc": wc,
        "mean_l": {ph: (l_sum[ph] / l_n[ph] if l_n[ph] else float("nan"))
                   for ph in counters},
        "snapshots": snaps,
        "halton_index_ranges": {"burnin": [halton_start, halton_start + burnin - 1],
                                "scored": [halton_start + burnin, halton_start + steps - 1]},
    }


def scored_estimates(ens, single, ref_mean=None, gold_mean=None):
    """Substrate estimator conventions (run_method) + gold MSE (V2-4)."""
    out = {}
    ens_c = ens - ens.mean(axis=0)[None, :]
    single_c = single - single.mean(axis=0)[None, :]
    out["trace_var"] = float(cp.mean(cp.sum(ens_c * ens_c, axis=1)).get())
    out["single_trace_var"] = float(cp.mean(cp.sum(single_c * single_c, axis=1)).get())
    if ref_mean is not None:
        ee = ens - ref_mean[None, :]
        se = single - ref_mean[None, :]
        out["mse"] = float(cp.mean(cp.sum(ee * ee, axis=1)).get())
        out["single_mse"] = float(cp.mean(cp.sum(se * se, axis=1)).get())
        out["bias_sq"] = float(cp.sum((ens.mean(axis=0) - ref_mean) ** 2).get())
    if gold_mean is not None:
        gm = cp.asarray(gold_mean, dtype=cp.float64)
        ge = ens.astype(cp.float64) - gm[None, :]
        out["gold_mse"] = float(cp.mean(cp.sum(ge * ge, axis=1)).get())
    return out


def load_gold():
    """V2-4 gold loader: exactly six kind='gold' rows, abort on duplicates;
    artifact sha256 recorded (and checked against the precommit pin)."""
    sha = full_sha256(GOLD_PATH)
    assert sha == GOLD_PIN, f"gold artifact sha256 {sha} != pinned {GOLD_PIN}"
    rows = [json.loads(line) for line in GOLD_PATH.open() if line.strip()]
    gold = {}
    n_gold = 0
    for r in rows:
        if r.get("kind") != "gold":
            continue
        n_gold += 1
        key = (r["dataset"], float(r["prior"]))
        assert key not in gold, f"duplicate gold row for {key}"
        gold[key] = {"gold_mean": np.asarray(r["gold_mean"], dtype=np.float64),
                     "gold_id": r["gold_id"]}
    assert n_gold == 6 and len(gold) == 6, f"expected exactly 6 gold rows, got {n_gold}"
    return gold, sha


# --------------------------------------------------------------------------
# Campaign driver
# --------------------------------------------------------------------------

TUNER_SPEC = {"adam_lr": ADAM_LR, "adam_beta1": ADAM_BETA1, "adam_beta2": ADAM_BETA2,
              "adam_stab": ADAM_STAB, "da_gamma": DA_GAMMA, "da_t0": DA_T0,
              "da_kappa": DA_KAPPA, "da_target": DA_TARGET, "alpha_floor": ALPHA_FLOOR,
              "l_max": L_MAX, "t_box": [T_BOX_LO, T_BOX_HI],
              "eps_box_factor": [EPS_BOX_LO, EPS_BOX_HI], "freeze_step": N_WARM,
              "n_equil": N_EQUIL, "n_score": N_SCORE, "ema_rate": EMA_RATE}


def design_fields(dataset, prior, replicate, dim, reps, eps0, pilot, quick):
    return {"target": "real_logistic_chees", "dataset": dataset, "prior": prior,
            "replicate": replicate, "dim": dim, "chains": K_CHAINS, "reps": reps,
            "n_ensembles": reps, "eps0_pilot": eps0, "pilot_accept": pilot,
            "eps_calibrated": True, "quick": bool(quick),
            "seed_base": seed_base(dataset, prior, replicate),
            "probe_seed": stream_seed(dataset, prior, replicate, "probe"),
            "tuner": TUNER_SPEC}


def closed_row(dfields, arm, coupling, res, dyn_seed, rot_seed, est, gold_info,
               variant="primary"):
    row = dict(dfields)
    hir = res["halton_index_ranges"]
    row.update({
        "kind": "closed_loop", "arm": arm, "coupling": coupling, "method": coupling,
        "variant": variant, "dyn_seed": dyn_seed, "rot_seed": rot_seed,
        "steps": hir["scored"][1], "n_warm": hir["warmup"][1],
        "n_equil": hir["equil"][1] - hir["equil"][0] + 1,
        "n_score": hir["scored"][1] - hir["scored"][0] + 1,
        "windows": res["windows"], "group_size": res["group_size"],
        "T_star": np.asarray(res["T_star"]).tolist(),
        "eps_star": np.asarray(res["eps_star"]).tolist(),
        "T_star_cf": (np.asarray(res["T_star_cf"]).tolist()
                      if res["T_star_cf"] is not None else None),
        "freeze_clamp": res["freeze_clamp"],
        "warmup_end_abar_mean": res["warmup_end_abar_mean"],
        "warmup_end_abar_median": res["warmup_end_abar_median"],
        "scored_abar_mean": res["scored_abar_mean"],
        "accept": res["accept"], "qc_counters": res["counters"],
        "ug": res["ug"], "bp": res["bp"], "wc": res["wc"], "mean_l": res["mean_l"],
        "m_probe": res["m_probe"], "m_probe_freeze": res["m_probe_freeze"],
        "mean_realized_tlen": res["mean_realized_tlen"],
        "frozenness_ok": res["frozenness_ok"],
        "tuner_traj": res["traj"],
        "halton_index_ranges": res["halton_index_ranges"],
        "elapsed_sec": sum(res["wc"].values()),
    })
    row.update(est)
    if gold_info is not None:
        row["gold_id"], row["gold_sha256"] = gold_info
    row["provenance"] = provenance()
    return row


def frozen_row(dfields, kind, arm, coupling, res, dyn_seed, rot_seed, est,
               gold_info, extra=None):
    row = dict(dfields)
    row.update({
        "kind": kind, "arm": arm, "coupling": coupling, "method": coupling,
        "variant": "primary", "dyn_seed": dyn_seed, "rot_seed": rot_seed,
        "windows": res["windows"], "accept": res["accept"],
        "qc_counters": res["counters"], "ug": res["ug"], "bp": res["bp"],
        "wc": res["wc"], "mean_l": res["mean_l"], "m_probe": res["m_probe"],
        "halton_index_ranges": res["halton_index_ranges"],
        "elapsed_sec": sum(res["wc"].values()),
    })
    row.update(est or {})
    if gold_info is not None:
        row["gold_id"], row["gold_sha256"] = gold_info
    if extra:
        row.update(extra)
    row["provenance"] = provenance()
    return row


def write_row(out_f, row):
    out_f.write(json.dumps(row, sort_keys=True) + "\n")
    out_f.flush()
    # Audit rec (a): during real (non-quick) runs the console brief suppresses
    # every estimate (trace_var/mse/gold_mse); raw JSONL and logs stay
    # embargoed until unblinding -- the --qc-only emitter is the only
    # pre-unblinding analysis surface (V2-10).
    keys = ["dataset", "prior", "replicate", "kind", "arm", "coupling",
            "variant", "freeze_clamp"]
    if row.get("quick"):
        keys += ["trace_var", "mse", "gold_mse"]
    brief = {k: row.get(k) for k in keys}
    if not row.get("quick"):
        brief["estimates"] = "EMBARGOED(rep-blinding; V2-10/audit rec a)"
    print(json.dumps(brief, sort_keys=True), flush=True)


def save_sidecar(path, res, u_rot, extra_arrays=None):
    payload = {"provenance": json.dumps(provenance(), sort_keys=True),
               "u_rot": np.asarray(u_rot)}
    if res is not None:
        payload.update({"T_star": np.asarray(res["T_star"]),
                        "eps_star": np.asarray(res["eps_star"])})
        if res.get("T_star_cf") is not None:
            payload["T_star_cf"] = np.asarray(res["T_star_cf"])
        if res.get("full_traj"):
            for k, vv in res["full_traj"].items():
                payload[f"traj_{k}"] = vv
    if extra_arrays:
        payload.update(extra_arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"refusing to overwrite sidecar {path}")
    np.savez_compressed(path, **payload)


def run_design(args, dataset, dim, prior, replicate, potential, eps0, pilot,
               gold, out_f, budgets):
    base = seed_base(dataset, prior, replicate)
    reps = budgets["reps"]
    quick = bool(args.quick)
    dfields = design_fields(dataset, prior, replicate, dim, reps, eps0, pilot, quick)
    probe = nrl.normalized_probe(dim, base + STREAMS["probe"]).get().astype(np.float64)
    u_closed = draw_rotations(base + STREAMS["rot_closed"], reps)
    u_replay = draw_rotations(base + STREAMS["rot_replay"], reps)
    u_ref = draw_rotations(base + STREAMS["rot_reference"], budgets["ref_reps"])
    gold_cell = gold[0].get((dataset, prior)) if (gold and gold[0]) else None
    gold_info = ((gold_cell["gold_id"], gold[1]) if gold_cell else None)
    gold_mean = gold_cell["gold_mean"] if gold_cell else None
    sidecar_dir = Path(args.sidecar_dir)
    tag = f"{dataset}_p{prior}_rep{replicate}"

    def closed(arm, coupling, dyn_off):
        dyn_seed = base + dyn_off
        res = run_closed_loop(potential, dim, reps, K_CHAINS, coupling, eps0,
                              dyn_seed, u_rot=u_closed, probe=probe,
                              n_warm=budgets["n_warm"], n_equil=budgets["n_equil"],
                              n_score=budgets["n_score"],
                              full_traj=args.log_full_tuner)
        return res, dyn_seed

    # Sequencing (section 2): arm A -> reference -> error pass -> arms B/C -> factorial
    res_a, dyn_a = closed("A", "iid", STREAMS["arm_A"])
    t_ref = float(np.median(res_a["T_star"]))
    eps_ref = float(np.median(res_a["eps_star"]))
    cp.random.seed(base + STREAMS["reference"])
    res_ref = run_frozen(potential, dim, budgets["ref_reps"], budgets["ref_chains"],
                         "iid", u_ref, np.full(budgets["ref_reps"], t_ref),
                         np.full(budgets["ref_reps"], eps_ref),
                         steps=budgets["ref_steps"], burnin=budgets["ref_burnin"])
    ref_mean = res_ref["ens"].mean(axis=0)
    write_row(out_f, closed_row(dfields, "A", "iid", res_a, dyn_a,
                                base + STREAMS["rot_closed"],
                                scored_estimates(res_a["ens"], res_a["single"],
                                                 ref_mean, gold_mean), gold_info))
    write_row(out_f, frozen_row(dfields, "reference", "reference", "iid", res_ref,
                                base + STREAMS["reference"],
                                base + STREAMS["rot_reference"],
                                scored_estimates(res_ref["ens"], res_ref["single"],
                                                 ref_mean, gold_mean), gold_info,
                                extra={"T_ref": t_ref, "eps_ref": eps_ref,
                                       "reps_ref": budgets["ref_reps"],
                                       "chains_ref": budgets["ref_chains"],
                                       "steps": budgets["ref_steps"],
                                       "burnin": budgets["ref_burnin"]}))
    if args.log_full_tuner:
        save_sidecar(sidecar_dir / f"chees_tuner_{tag}_armA.npz", res_a, u_closed)
    res_by_arm = {"A": res_a}
    for arm, coupling, off in (("B", "zero_sum", STREAMS["arm_B"]),
                               ("C", "antithetic", STREAMS["arm_C"])):
        res, dyn = closed(arm, coupling, off)
        res_by_arm[arm] = res
        write_row(out_f, closed_row(dfields, arm, coupling, res, dyn,
                                    base + STREAMS["rot_closed"],
                                    scored_estimates(res["ens"], res["single"],
                                                     ref_mean, gold_mean), gold_info))
        if args.log_full_tuner:
            save_sidecar(sidecar_dir / f"chees_tuner_{tag}_arm{arm}.npz", res, u_closed)
    # Replay 2x2 factorial (V2-5): shared x0 from +240,000, matched jitter
    cp.random.seed(base + STREAMS["replay_x0"])
    x0 = cp.random.standard_normal((reps, K_CHAINS, dim), dtype=cp.float32)
    for src, coupling in REPLAY_RUNS:
        name = f"replay_{src}_{coupling}"
        dyn_seed = base + STREAMS[name]
        cp.random.seed(dyn_seed)
        res = run_frozen(potential, dim, reps, K_CHAINS, coupling, u_replay,
                         np.asarray(res_by_arm[src]["T_star"], dtype=np.float64),
                         np.asarray(res_by_arm[src]["eps_star"], dtype=np.float64),
                         steps=budgets["replay_total"], burnin=budgets["replay_burn"],
                         x0=x0)
        write_row(out_f, frozen_row(
            dfields, "replay", name, coupling, res, dyn_seed,
            base + STREAMS["rot_replay"],
            scored_estimates(res["ens"], res["single"], ref_mean, gold_mean),
            gold_info, extra={"source_arm": src,
                              "x0_seed": base + STREAMS["replay_x0"],
                              "steps": budgets["replay_total"],
                              "burnin": budgets["replay_burn"]}))
    save_sidecar(sidecar_dir / f"chees_design_{tag}.npz", None, u_closed,
                 extra_arrays={"u_replay": u_replay, "u_ref": u_ref, "probe": probe,
                               "x0": x0.get()})
    # NOTE: the G=4 secondary variant is TRIGGER-ONLY (V2-2) and runs as a
    # SEPARATE SECOND STAGE (--g4-stage) after the arm-A instability trigger
    # is confirmed from the completed primary rows (audit finding 5); it is
    # deliberately NOT runnable alongside the primary campaign.


def campaign(args):
    # Fix-verification defect 1: priors preflight -- unique and registered,
    # BEFORE any calibration (applies to quick too; unregistered priors have
    # no seed block).
    priors = args.priors or PRIORS
    if len(set(priors)) != len(priors) or not set(priors).issubset(set(PRIORS)):
        raise SystemExit(f"priors must be unique registered values from "
                         f"{PRIORS}; got {priors}")
    if not args.quick:
        # Fix-verification defect 2: production execution refuses the stub.
        if IS_STUB:
            raise SystemExit("PRODUCTION REFUSED: non-quick campaign execution "
                             "under the numpy cupy stub is forbidden (audit "
                             "fix-verification defect 2); run on real cupy/GPU")
        if args.replicate not in (0, 1, 2):
            raise SystemExit(f"non-quick replicate must be in {{0,1,2}}; got "
                             f"{args.replicate} (fix-verification defect 1)")
        # Audit finding 6: registered-design overrides are quick-only.
        if args.reps is not None:
            raise SystemExit("--reps is a quick-only override; the registered "
                             "design fixes ensembles at 128/96/48 (V2-7/audit 6)")
        if args.chunk_rows != 1024:
            raise SystemExit("--chunk-rows is a quick-only override; the "
                             "registered convention is 1024")
        if not args.log_full_tuner:
            # V2-N + audit finding 4: full tuner sidecars are MANDATORY for
            # every non-quick replicate.
            print("note: enabling --log-full-tuner (mandatory for non-quick "
                  "runs; V2-N/audit finding 4)")
            args.log_full_tuner = True
    budgets = {"n_warm": N_WARM, "n_equil": N_EQUIL, "n_score": N_SCORE,
               "replay_total": REPLAY_TOTAL, "replay_burn": REPLAY_BURN,
               "ref_steps": REF_STEPS, "ref_burnin": REF_BURN,
               "ref_reps": REF_REPS, "ref_chains": REF_CHAINS}
    if args.quick:
        budgets.update({"n_warm": 45, "n_equil": 5, "n_score": 15,
                        "replay_total": 30, "replay_burn": 15,
                        "ref_steps": 60, "ref_burnin": 30,
                        "ref_reps": 8, "ref_chains": 8})
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    priors = args.priors or PRIORS
    gold = load_gold() if not args.quick else (None, None)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as out_f:
        for dataset in datasets:
            data_x, data_y = nrl.load_dataset(args.data_dir, dataset)
            dim = int(data_x.shape[1])
            for prior in priors:
                eps0, pilot = nrl.calibrate_eps(dataset, dim, prior, data_x, data_y,
                                                args.chunk_rows)
                potential = nrl.logistic_potential(data_x, data_y, prior,
                                                   args.chunk_rows)
                budgets_d = dict(budgets)
                budgets_d["reps"] = args.reps or CELL_REPS[dataset]
                if args.quick:
                    budgets_d["reps"] = args.reps or 8
                run_design(args, dataset, dim, prior, args.replicate, potential,
                           eps0, pilot, gold, out_f, budgets_d)


def group_means(arr, group=G4_GROUP):
    """(reps, dim) -> (reps//group, dim) four-ensemble group means (V2-2)."""
    r, d = arr.shape
    return arr.reshape(r // group, group, d).mean(axis=1)


def g4_stage(args):
    """Audit finding 5: the TRIGGER-ONLY G=4 second stage.  Loads and
    validates ALL primary rows against the lock manifest, confirms the arm-A
    median-IQR trigger (V2-2) from them, and only then runs G=4 arms A/B/C
    for every cell/replicate (streams +500k/510k/520k, rotations +430k).
    Location/probe statistics come from four-ensemble GROUP means and
    schedule statistics from group scalars, labeled clustered n = reps/4.
    The primary H1 verdict remains UNRESOLVED (K1); G=4 is archived context."""
    rows = load_rows(args.table_inputs)
    manifest, gold_map, gold_sha, _div = validate_campaign(
        rows, args.lock_manifest, args.sidecar_dir, args.selftest_audit_dir,
        full=True, g4_expected="forbid")
    prim = [r for r in rows if r["variant"] == "primary"]
    trigger = []
    for ds in DATASETS:
        for prior in PRIORS:
            iqrs = [iqr_log2(r["T_star"]) for r in prim
                    if (r["dataset"], r["prior"], r["kind"], r["arm"])
                    == (ds, prior, "closed_loop", "A")]
            assert len(iqrs) == 3
            m = float(np.median(iqrs))
            if np.isfinite(m) and m > 1.0:
                trigger.append((ds, prior, m))
    if not trigger:
        raise SystemExit(
            "G=4 stage refused: the arm-A M=8 instability trigger (median-over-"
            "replicates IQR_e(log2 T*_A) > 1.0) fired in NO cell; the G=4 "
            "secondary variant is trigger-only (V2-2)")
    print(f"K1 TRIGGER CONFIRMED from primary rows: {trigger}")
    print("Primary H1 verdict = UNRESOLVED (K1, V2-2); running the G=4 "
          "secondary variant for ALL cells/replicates as archived context.")
    # Fix-verification defect 2: G4 EXECUTION refuses the stub (placed after
    # the validation/trigger logic so that logic stays CPU-testable).
    if IS_STUB:
        raise SystemExit("PRODUCTION REFUSED: --g4-stage execution under the "
                         "numpy cupy stub is forbidden (fix-verification "
                         "defect 2); run on real cupy/GPU")
    if args.out is None or args.out.exists():
        raise SystemExit("--g4-stage requires a fresh --out path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sidecar_dir = Path(args.sidecar_dir)
    with args.out.open("x", encoding="utf-8") as out_f:
        for dataset in DATASETS:
            data_x, data_y = nrl.load_dataset(args.data_dir, dataset)
            dim = int(data_x.shape[1])
            reps = CELL_REPS[dataset]
            for prior in PRIORS:
                eps0, pilot = nrl.calibrate_eps(dataset, dim, prior, data_x,
                                                data_y, 1024)
                # Judgment-call-5 addendum: the recomputed pilot must match
                # every corresponding primary row exactly before any G4 work.
                for r in prim:
                    if (r["dataset"], r["prior"]) != (dataset, prior):
                        continue
                    assert r["eps0_pilot"] == eps0 and r["pilot_accept"] == pilot, (
                        f"G4-stage pilot mismatch at {dataset}/{prior}/rep"
                        f"{r['replicate']}/{r['arm']}: recomputed ({eps0},"
                        f"{pilot}) != primary row ({r['eps0_pilot']},"
                        f"{r['pilot_accept']})")
                potential = nrl.logistic_potential(data_x, data_y, prior, 1024)
                gold_cell = gold_map[(dataset, prior)]
                for rep in range(3):
                    base = seed_base(dataset, prior, rep)
                    dfields = design_fields(dataset, prior, rep, dim, reps,
                                            eps0, pilot, False)
                    probe = nrl.normalized_probe(
                        dim, base + STREAMS["probe"]).get().astype(np.float64)
                    u_g, u_f = draw_rotations_g4(base + STREAMS["rot_g4"],
                                                 reps // G4_GROUP, reps)
                    tag = f"{dataset}_p{prior}_rep{rep}"
                    for arm, coupling, off in (("A", "iid", STREAMS["g4_A"]),
                                               ("B", "zero_sum", STREAMS["g4_B"]),
                                               ("C", "antithetic", STREAMS["g4_C"])):
                        dyn = base + off
                        res = run_closed_loop(
                            potential, dim, reps, K_CHAINS, coupling, eps0, dyn,
                            u_rot=u_g, group_size=G4_GROUP, u_fork=u_f,
                            probe=probe, crossfit=False, full_traj=True)
                        # clustered estimates from four-ensemble group means
                        est = scored_estimates(group_means(res["ens"]),
                                               group_means(res["single"]),
                                               None, gold_cell["gold_mean"])
                        per_ens = scored_estimates(res["ens"], res["single"])
                        est["trace_var_per_ensemble"] = per_ens["trace_var"]
                        row = closed_row(dfields, arm, coupling, res, dyn,
                                         base + STREAMS["rot_g4"], est,
                                         (gold_cell["gold_id"], gold_sha),
                                         variant="g4")
                        row["g4_clustered"] = True
                        row["cluster_label"] = f"clustered, n={reps // G4_GROUP}"
                        write_row(out_f, row)
                        save_sidecar(sidecar_dir / f"chees_tuner_{tag}_arm{arm}_g4.npz",
                                     res, u_g, extra_arrays={"u_fork": u_f})
    print(f"G=4 stage complete -> {args.out} (context only; clustered n=reps/4)")


# --------------------------------------------------------------------------
# --print-plan: full run matrix for the code audit
# --------------------------------------------------------------------------

def print_plan():
    th = triple_hash()
    print("== ChEES closed-loop campaign plan (locked precommit "
          f"{th['precommit_sha256'][:16]}...) ==")
    print(f"chees_script_sha256   = {th['chees_script_sha256']}")
    print(f"nuts_substrate_sha256 = {th['nuts_substrate_sha256']}")
    print(f"precommit_sha256      = {th['precommit_sha256']}")
    print(f"phases: warmup 1-{N_WARM} (adapt), freeze@{N_WARM}, equil "
          f"{N_WARM + 1}-{N_WARM + N_EQUIL}, scored {N_WARM + N_EQUIL + 1}-{N_TOTAL}; "
          f"replay {REPLAY_BURN}+{REPLAY_SCORE}; reference {REF_STEPS} "
          f"(burnin {REF_BURN}, {REF_REPS}x{REF_CHAINS})")
    total_runs = 0
    total_outer = 0
    total_walker_steps = 0
    for ds in DATASETS:
        reps = CELL_REPS[ds]
        for prior in PRIORS:
            for rep in range(3):
                base = seed_base(ds, prior, rep)
                print(f"-- cell {ds}/prior={prior} replicate {rep}: base {base} "
                      f"reps={reps} K={K_CHAINS} probe={base + STREAMS['probe']}")
                for arm, coupling in ARMS:
                    dyn = base + STREAMS[f"arm_{arm}"]
                    print(f"   closed_loop arm {arm} ({coupling:>10s}): dyn {dyn} "
                          f"rot {base + STREAMS['rot_closed']} steps {N_TOTAL} "
                          f"halton 1-{N_TOTAL}")
                    total_runs += 1
                    total_outer += N_TOTAL
                    total_walker_steps += N_TOTAL * reps * K_CHAINS
                print(f"   reference (iid, {REF_REPS}x{REF_CHAINS} @ arm-A median "
                      f"T*/eps*): dyn {base + STREAMS['reference']} rot "
                      f"{base + STREAMS['rot_reference']} steps {REF_STEPS} "
                      f"halton 1-{REF_STEPS}")
                total_runs += 1
                total_outer += REF_STEPS
                total_walker_steps += REF_STEPS * REF_REPS * REF_CHAINS
                for src, coupling in REPLAY_RUNS:
                    name = f"replay_{src}_{coupling}"
                    print(f"   {name:>22s}: dyn {base + STREAMS[name]} rot "
                          f"{base + STREAMS['rot_replay']} x0 "
                          f"{base + STREAMS['replay_x0']} steps {REPLAY_TOTAL} "
                          f"halton 1-{REPLAY_TOTAL}")
                    total_runs += 1
                    total_outer += REPLAY_TOTAL
                    total_walker_steps += REPLAY_TOTAL * reps * K_CHAINS
                print(f"   [trigger-only G=4 SECOND STAGE (--g4-stage): arms "
                      f"A/B/C dyn {base + STREAMS['g4_A']}/{base + STREAMS['g4_B']}/"
                      f"{base + STREAMS['g4_C']} rot {base + STREAMS['rot_g4']} "
                      f"(reserved {base + STREAMS['g4_reserved']}); 650 steps; "
                      f"runs only after the arm-A trigger is confirmed from "
                      f"completed primary rows]")
    print(f"== totals (primary, excl. trigger-only G=4): {total_runs} runs, "
          f"{total_outer} outer steps, {total_walker_steps} walker outer steps ==")
    print(f"selftest seeds: {SELFTEST_SEEDS}")
    print(f"archived read-only replays (S-7): {ARCHIVED_READONLY}")
    s10_seed_uniqueness()


# --------------------------------------------------------------------------
# Emitter: QC gates, --qc-only (V2-10 blinding), full table + ladder
# --------------------------------------------------------------------------

def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def expected_seeds(row):
    ds, prior, rep = row["dataset"], row["prior"], row["replicate"]
    kind, arm = row["kind"], row["arm"]
    if kind == "closed_loop":
        variant = row.get("variant", "primary")
        dyn = stream_seed(ds, prior, rep, (f"arm_{arm}" if variant == "primary"
                                           else f"g4_{arm}"))
        rot = stream_seed(ds, prior, rep, ("rot_closed" if variant == "primary"
                                           else "rot_g4"))
    elif kind == "replay":
        dyn = stream_seed(ds, prior, rep, arm)
        rot = stream_seed(ds, prior, rep, "rot_replay")
    elif kind == "reference":
        dyn = stream_seed(ds, prior, rep, "reference")
        rot = stream_seed(ds, prior, rep, "rot_reference")
    else:
        raise AssertionError(f"unknown kind {kind}")
    return dyn, rot


def expected_row_shape(row):
    """Registered per-kind budgets and Halton index ranges (audit finding 6)."""
    kind = row["kind"]
    if kind == "closed_loop":
        return {"halton_index_ranges": {"warmup": [1, N_WARM],
                                        "equil": [N_WARM + 1, N_WARM + N_EQUIL],
                                        "scored": [N_WARM + N_EQUIL + 1, N_TOTAL]},
                "n_warm": N_WARM, "n_equil": N_EQUIL, "n_score": N_SCORE,
                "steps": N_TOTAL, "windows": N_SCORE}
    if kind == "replay":
        return {"halton_index_ranges": {"burnin": [1, REPLAY_BURN],
                                        "scored": [REPLAY_BURN + 1, REPLAY_TOTAL]},
                "steps": REPLAY_TOTAL, "burnin": REPLAY_BURN,
                "windows": REPLAY_SCORE}
    return {"halton_index_ranges": {"burnin": [1, REF_BURN],
                                    "scored": [REF_BURN + 1, REF_STEPS]},
            "steps": REF_STEPS, "burnin": REF_BURN,
            "windows": REF_STEPS - REF_BURN}


EXPECTED_PRIMARY_KINDS = sorted(
    [("closed_loop", "A"), ("closed_loop", "B"), ("closed_loop", "C"),
     ("reference", "reference")]
    + [("replay", f"replay_{s}_{c}") for s, c in REPLAY_RUNS])
ARM_COUPLING = dict(ARMS)


def _json_eq(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


CANON_PRIMARY_KINDS = set(EXPECTED_PRIMARY_KINDS)


def canonical_key(row):
    """Fix-verification defect 1: exact canonical row-key mapping.  Every row
    must map to a registered key; unknown variant/kind/arm/dataset/prior/
    replicate is an ABORT, never silently filtered."""
    ds, prior, rep = row.get("dataset"), row.get("prior"), row.get("replicate")
    variant = row.get("variant")
    kind, arm = row.get("kind"), row.get("arm")
    assert ds in DATASETS, f"unregistered dataset {ds!r}"
    assert prior in PRIORS, f"unregistered prior {prior!r}"
    assert rep in (0, 1, 2), f"unregistered replicate {rep!r}"
    assert variant in ("primary", "g4"), (
        f"unregistered variant {variant!r} at {ds}/{prior}/rep{rep}/{arm} "
        f"(canonical row-key validation)")
    if variant == "g4":
        assert kind == "closed_loop" and arm in ("A", "B", "C"), (
            f"unregistered G4 row (kind={kind!r}, arm={arm!r})")
    else:
        assert (kind, arm) in CANON_PRIMARY_KINDS, (
            f"unregistered primary (kind, arm) = ({kind!r}, {arm!r})")
    return (ds, prior, rep, kind, arm, variant)


def _schema_array(row, field, length, loc, allow_none=False):
    """Schema-vs-divergence separation (defect 3): a MISSING or wrong-shape
    mandatory array is corruption -> ABORT; values are checked separately."""
    assert field in row, f"SCHEMA ABORT at {loc}: mandatory field {field} missing"
    val = row[field]
    if val is None:
        assert allow_none, f"SCHEMA ABORT at {loc}: {field} is None"
        return None
    arr = np.asarray(val, dtype=np.float64)
    assert arr.shape == (length,), (
        f"SCHEMA ABORT at {loc}: {field} shape {arr.shape} != ({length},)")
    return arr


def _flag_nonfinite(arr, positive, loc, field, diverged, key4):
    """Nonfinite (or nonpositive where required) sampler values -> DIVERGED
    flag (V2-9 result), never an abort.  key4 = (ds, prior, rep, variant):
    closing-check residual 1 -- the variant travels with every record so
    G4-only divergence never contaminates the primary ladder."""
    if arr is None:
        return
    bad = ~np.isfinite(arr)
    if positive:
        bad |= arr <= 0.0
    if np.any(bad):
        diverged.append((*key4, f"nonfinite/nonpositive {field} at {loc}"))


def validate_campaign(rows, lock_path, sidecar_dir, audit_dir, full=True,
                      g4_expected="auto"):
    """Audit findings 4 + 6 + fix-verification defects 1-3: manifest-bound
    validation of rows, sidecars, and S-7 audit artifacts against the
    IMMUTABLE executable-lock manifest (never the currently executing bytes);
    canonical row-key matrix (unknown variants/kinds abort; exactly 144
    primary rows in full mode; exactly 54 G4 rows IFF the K1 trigger fired
    when g4_expected='auto'; g4_expected='forbid' for the pre-G4 stage);
    production backend provenance enforced everywhere; schema corruption
    aborts while sampler-nonfinite values return DIVERGED flags (V2-9)."""
    manifest = load_lock_manifest(lock_path)
    live = triple_hash()
    for key in HASH_KEYS:
        assert live[key] == manifest[key], (
            f"LOCK MISMATCH: on-disk {key} {live[key][:16]}... != manifest "
            f"{manifest[key][:16]}... -- V2-9 defect procedure applies")
    gold, gold_sha = load_gold()
    assert gold_sha == GOLD_PIN
    diverged = []
    eps0_seen = {}
    keys_seen = []
    for r in rows:
        key = canonical_key(r)                      # unknown key -> ABORT
        keys_seen.append(key)
        ds, prior, rep = key[0], key[1], key[2]
        variant = key[5]
        loc = f"{ds}/{prior}/rep{rep}/{r['arm']}[{variant}]"
        key4 = (ds, prior, rep, variant)
        assert r.get("quick") is False, f"{loc}: quick is not False"
        assert r.get("target") == "real_logistic_chees", f"{loc}: target field"
        assert r.get("method") == r.get("coupling"), f"{loc}: method != coupling"
        assert r.get("eps_calibrated") is True, f"{loc}: calibration status"
        assert (isinstance(r.get("eps0_pilot"), float)
                and np.isfinite(r["eps0_pilot"]) and r["eps0_pilot"] > 0), \
            f"{loc}: eps0_pilot invalid"
        # closing-check residual 3: pilot_accept must be a valid rate
        assert (isinstance(r.get("pilot_accept"), float)
                and np.isfinite(r["pilot_accept"])
                and 0.0 <= r["pilot_accept"] <= 1.0), \
            f"{loc}: pilot_accept invalid (must be finite in [0, 1])"
        assert r.get("dim") == len(gold[(ds, float(prior))]["gold_mean"]), \
            f"{loc}: dimension does not match the gold artifact"
        pv = r["provenance"]
        for hk in HASH_KEYS:
            assert pv[hk] == manifest[hk], (
                f"hash-identity abort at {loc}: {hk} {pv[hk][:16]}... != "
                f"manifest {manifest[hk][:16]}...")
        assert pv.get("cupy_backend") == PRODUCTION_BACKEND, (
            f"BACKEND ABORT at {loc}: cupy_backend={pv.get('cupy_backend')!r} "
            f"!= '{PRODUCTION_BACKEND}' (production rows must come from real "
            f"cupy; fix-verification defect 2)")
        base = seed_base(ds, prior, rep)
        assert r["seed_base"] == base, f"seed_base mismatch at {loc}"
        assert r["probe_seed"] == base + STREAMS["probe"], f"probe_seed at {loc}"
        dyn, rot = expected_seeds(r)
        assert r["dyn_seed"] == dyn and r["rot_seed"] == rot, (
            f"seed-map abort at {loc}: recorded ({r['dyn_seed']},{r['rot_seed']})"
            f" != expected ({dyn},{rot})")
        for k, v in expected_row_shape(r).items():
            assert _json_eq(r.get(k), v), f"{loc}: registered {k} mismatch: {r.get(k)} != {v}"
        assert r["chains"] == K_CHAINS, f"{loc}: chains"
        assert r["reps"] == CELL_REPS[ds] and r["n_ensembles"] == r["reps"], \
            f"{loc}: registered ensemble count violated"
        assert _json_eq(r["tuner"], TUNER_SPEC), f"{loc}: tuner constants drifted"
        assert r.get("gold_sha256") == GOLD_PIN, f"{loc}: gold artifact hash"
        assert r.get("gold_id") == gold[(ds, float(prior))]["gold_id"], \
            f"{loc}: gold_id mismatch"
        eps0_seen.setdefault((ds, prior, rep), set()).add(
            (r["eps0_pilot"], r["pilot_accept"]))  # residual 3: pair consistency
        if r["kind"] == "closed_loop":
            assert ARM_COUPLING.get(r["arm"]) == r["coupling"], f"{loc}: arm/coupling"
            assert r.get("frozenness_ok") is True, f"{loc}: frozenness not asserted"
            reps = r["reps"]
            is_g4 = variant == "g4"
            assert r.get("group_size") == (G4_GROUP if is_g4 else 1), \
                f"{loc}: group_size"
            n_units_exp = reps // G4_GROUP if is_g4 else reps
            # SCHEMA: mandatory arrays present with registered shapes (abort);
            # SAMPLER: nonfinite/nonpositive values -> DIVERGED (V2-9 result)
            for field, positive in (("T_star", True), ("eps_star", True)):
                arr = _schema_array(r, field, n_units_exp, loc)
                _flag_nonfinite(arr, positive, loc, field, diverged, key4)
            cf = _schema_array(r, "T_star_cf", reps, loc, allow_none=is_g4)
            if is_g4:
                assert cf is None, f"{loc}: G4 rows must have T_star_cf=None"
            else:
                _flag_nonfinite(cf, True, loc, "T_star_cf", diverged, key4)
            for field in ("m_probe", "m_probe_freeze"):
                arr = _schema_array(r, field, reps, loc)
                _flag_nonfinite(arr, False, loc, field, diverged, key4)
            tlen = _schema_array(r, "mean_realized_tlen", reps, loc)
            _flag_nonfinite(tlen, True, loc, "mean_realized_tlen", diverged, key4)
            tt = r.get("tuner_traj")
            assert isinstance(tt, dict) and all(k in tt for k in TRAJ_KEYS), \
                f"SCHEMA ABORT at {loc}: tuner_traj keys"
            assert tt["step"] == TRAJ_STEPS_REGISTERED, \
                f"SCHEMA ABORT at {loc}: tuner_traj step grid"
            for k in TRAJ_KEYS[1:]:
                assert len(tt[k]) == len(TRAJ_STEPS_REGISTERED), \
                    f"SCHEMA ABORT at {loc}: tuner_traj {k} length"
                vals = np.asarray(tt[k], dtype=np.float64)
                if k.startswith("dghat") and is_g4:
                    continue        # structurally NaN (no cross-fit in G4)
                _flag_nonfinite(vals, k[0] in ("T", "e"), loc,
                                f"tuner_traj.{k}", diverged, key4)
        elif r["kind"] == "replay":
            assert r["x0_seed"] == base + STREAMS["replay_x0"], f"{loc}: x0_seed"
            assert r["arm"] == f"replay_{r['source_arm']}_{r['coupling']}", \
                f"{loc}: replay identity"
        elif r["kind"] == "reference":
            assert r["coupling"] == "iid", f"{loc}: reference coupling"
            assert r["reps_ref"] == REF_REPS and r["chains_ref"] == REF_CHAINS, \
                f"{loc}: reference shape"
        for field in ("trace_var", "mse", "gold_mse"):
            val = r.get(field)
            if val is not None and (not np.isfinite(val) or val < 0
                                    or (field == "trace_var" and val == 0.0)):
                diverged.append((*key4, f"{field}={val} at {loc}"))
    from collections import Counter
    dups = [k for k, c in Counter(keys_seen).items() if c > 1]
    assert not dups, f"duplicate canonical row keys: {dups}"
    for dkey, vals in eps0_seen.items():
        assert len(vals) == 1, (f"(eps0_pilot, pilot_accept) inconsistent within "
                                f"design {dkey}: {vals}")
    # Canonical row matrix (defect 1): the loaded key set must be EXACTLY the
    # registered matrix -- no extras, omissions, or duplicates possible.
    prim = [r for r in rows if r["variant"] == "primary"]
    g4 = [r for r in rows if r["variant"] == "g4"]
    prim_keys = {canonical_key(r) for r in prim}
    designs = sorted({k[:3] for k in keys_seen})
    for d in designs:
        sub = sorted((k[3], k[4]) for k in prim_keys if k[:3] == d)
        assert sub == EXPECTED_PRIMARY_KINDS, f"row-matrix abort {d}: got {sub}"
        g4sub = sorted(k[4] for k in {canonical_key(r) for r in g4}
                       if k[:3] == d)
        assert g4sub in ([], ["A", "B", "C"]), f"G4 row-matrix abort {d}: {g4sub}"
    if full:
        expected_primary = {(ds, prior, rep, kind, arm, "primary")
                            for ds in DATASETS for prior in PRIORS
                            for rep in range(3)
                            for kind, arm in EXPECTED_PRIMARY_KINDS}
        assert prim_keys == expected_primary and len(prim) == 144, (
            f"canonical primary matrix abort: {len(prim)} rows; "
            f"missing={sorted(expected_primary - prim_keys)[:4]} "
            f"extra={sorted(prim_keys - expected_primary)[:4]}")
        # G4 conditional completeness (defect 1): exactly 54 rows IFF the
        # arm-A K1 trigger fired on the primary rows; zero otherwise.
        trig = False
        for ds in DATASETS:
            for prior in PRIORS:
                iqrs = [iqr_log2(r["T_star"]) for r in prim
                        if (r["dataset"], r["prior"], r["kind"], r["arm"])
                        == (ds, prior, "closed_loop", "A")]
                m = float(np.median(iqrs)) if len(iqrs) == 3 else float("nan")
                if np.isfinite(m) and m > 1.0:
                    trig = True
        if g4_expected == "forbid":
            assert len(g4) == 0, (
                f"G4 rows present in a pre-G4-stage validation ({len(g4)})")
        elif trig:
            assert len(g4) == 54, (
                f"K1 trigger FIRED on primary rows: exactly 54 G4 context rows "
                f"are REQUIRED, got {len(g4)} (mandatory triggered context)")
        else:
            assert len(g4) == 0, (
                f"{len(g4)} G4 rows present but the K1 trigger did NOT fire: "
                f"untriggered G4 artifacts are rejected (V2-2)")
    validate_sidecars(rows, sidecar_dir, manifest, diverged, strict=full)
    for name in S7_AUDIT_ARTIFACTS:
        path = Path(audit_dir) / name
        assert path.exists(), (
            f"S-7 audit artifact missing: {path} (all three must come from ONE "
            f"final GPU --selftest run under the locked triple hash)")
        with np.load(path, allow_pickle=True) as dd:
            spv = json.loads(str(dd["provenance"]))
            for key in HASH_KEYS:
                assert spv[key] == manifest[key], (
                    f"S-7 artifact {name}: {key} does not match the lock manifest")
            # Fix-verification defect 2: backend provenance, never filenames --
            # a renamed CPU-stub artifact fails here.
            assert spv.get("cupy_backend") == PRODUCTION_BACKEND, (
                f"S-7 artifact {name}: BACKEND ABORT -- cupy_backend="
                f"{spv.get('cupy_backend')!r} != '{PRODUCTION_BACKEND}' (the "
                f"registered artifacts must come from the one-run GPU suite)")
            assert "p0" in dd.files, f"S-7 artifact {name}: p0 archive missing"
            p0 = dd["p0"]
            assert p0.dtype == np.float32, f"S-7 artifact {name}: p0 dtype {p0.dtype}"
            assert p0.shape == S7_P0_SHAPES[name], (
                f"S-7 artifact {name}: p0 shape {p0.shape} != "
                f"{S7_P0_SHAPES[name]}")
            assert np.all(np.isfinite(p0)), f"S-7 artifact {name}: nonfinite p0"
    s10_seed_uniqueness(verbose=False)
    return manifest, gold, gold_sha, diverged


def validate_sidecars(rows, sidecar_dir, manifest, diverged, strict=True):
    """Audit finding 4 + fix-verification defects 2/3: full tuner sidecars are
    mandatory for every closed-loop row; missing/extra/duplicate or
    schema-mismatched sidecars ABORT; production backend provenance is
    enforced; row-vs-sidecar equality is NaN-aware (identical sampler NaNs
    are equal, not corruption); nonfinite trajectory VALUES flag DIVERGED.
    strict=True (full table): the sidecar set must EQUAL the expected set.
    strict=False (subset --qc-only mid-campaign): every expected sidecar must
    exist and validate; files belonging to replicates not in `rows` are
    tolerated with a warning (later replicates legitimately share the dir)."""
    sidecar_dir = Path(sidecar_dir)
    expected = {}
    design_expected = {}
    for r in rows:
        ds, prior, rep = r["dataset"], r["prior"], r["replicate"]
        tag = f"{ds}_p{prior}_rep{rep}"
        design_expected[f"chees_design_{tag}.npz"] = (ds, prior, rep)
        if r["kind"] != "closed_loop":
            continue
        g4 = r.get("variant") == "g4"
        name = f"chees_tuner_{tag}_arm{r['arm']}{'_g4' if g4 else ''}.npz"
        assert name not in expected, f"duplicate closed-loop row for sidecar {name}"
        expected[name] = r
    found = sorted(p.name for p in sidecar_dir.glob("chees_tuner_*.npz"))
    found_d = sorted(p.name for p in sidecar_dir.glob("chees_design_*.npz"))
    if strict:
        assert found == sorted(expected), (
            f"sidecar set mismatch (missing/extra/duplicate):\n expected "
            f"{sorted(expected)}\n found {found}")
        assert found_d == sorted(design_expected), (
            f"design-sidecar set mismatch:\n expected {sorted(design_expected)}\n "
            f"found {found_d}")
    else:
        missing = sorted(set(expected) - set(found))
        missing_d = sorted(set(design_expected) - set(found_d))
        assert not missing and not missing_d, (
            f"mandatory sidecars missing: {missing + missing_d}")
        extra = sorted((set(found) - set(expected))
                       | (set(found_d) - set(design_expected)))
        if extra:
            print(f"  note: {len(extra)} sidecar(s) outside the loaded row "
                  f"subset (other replicates); validated set covers loaded "
                  f"rows only")
    for name, r in expected.items():
        with np.load(sidecar_dir / name, allow_pickle=True) as d:
            spv = json.loads(str(d["provenance"]))
            for key in HASH_KEYS:
                assert spv[key] == manifest[key], f"sidecar {name}: {key} mismatch"
            assert spv.get("cupy_backend") == PRODUCTION_BACKEND, (
                f"sidecar {name}: BACKEND ABORT -- cupy_backend="
                f"{spv.get('cupy_backend')!r} != '{PRODUCTION_BACKEND}'")
            key4 = (r["dataset"], r["prior"], r["replicate"],
                    r.get("variant", "primary"))
            # NaN-aware equality (defect 3): identical sampler NaNs are equal
            assert np.array_equal(d["T_star"],
                                  np.asarray(r["T_star"], dtype=np.float64),
                                  equal_nan=True), f"sidecar {name}: T_star != row"
            assert np.array_equal(d["eps_star"],
                                  np.asarray(r["eps_star"], dtype=np.float64),
                                  equal_nan=True), f"sidecar {name}: eps_star != row"
            base = seed_base(r["dataset"], r["prior"], r["replicate"])
            reps = r["reps"]
            is_g4 = r.get("variant") == "g4"
            if is_g4:
                u_g, u_f = draw_rotations_g4(base + STREAMS["rot_g4"],
                                             reps // G4_GROUP, reps)
                assert np.array_equal(d["u_rot"], u_g), f"{name}: u_rot(groups)"
                assert np.array_equal(d["u_fork"], u_f), f"{name}: u_fork"
            else:
                u_c = draw_rotations(base + STREAMS["rot_closed"], reps)
                assert np.array_equal(d["u_rot"], u_c), f"{name}: u_rot != stream"
            n_units = (reps // G4_GROUP if is_g4 else reps)
            for k in ("traj_T", "traj_eps", "traj_ghat", "traj_ghat_cf",
                      "traj_abar"):
                assert k in d.files and d[k].shape == (N_WARM, n_units), \
                    f"SCHEMA ABORT sidecar {name}: {k} missing or wrong shape"
                if k == "traj_ghat_cf" and is_g4:
                    continue        # structurally NaN (no cross-fit in G4)
                if not np.all(np.isfinite(d[k])):
                    diverged.append((*key4,
                                     f"nonfinite sidecar trajectory {k} in {name}"))
    for name, (ds, prior, rep) in design_expected.items():
        with np.load(sidecar_dir / name, allow_pickle=True) as d:
            spv = json.loads(str(d["provenance"]))
            for key in HASH_KEYS:
                assert spv[key] == manifest[key], f"design sidecar {name}: {key}"
            assert spv.get("cupy_backend") == PRODUCTION_BACKEND, (
                f"design sidecar {name}: BACKEND ABORT -- cupy_backend="
                f"{spv.get('cupy_backend')!r}")
            base = seed_base(ds, prior, rep)
            reps = CELL_REPS[ds]
            assert np.array_equal(d["u_rot"],
                                  draw_rotations(base + STREAMS["rot_closed"], reps))
            assert np.array_equal(d["u_replay"],
                                  draw_rotations(base + STREAMS["rot_replay"], reps))
            assert np.array_equal(d["u_ref"],
                                  draw_rotations(base + STREAMS["rot_reference"],
                                                 REF_REPS))
            assert d["x0"].shape[0] == reps and d["x0"].shape[1] == K_CHAINS
            pr = d["probe"]
            assert pr.ndim == 1 and np.all(np.isfinite(pr)) and \
                abs(float(np.linalg.norm(pr)) - 1.0) < 1e-5 and pr[0] == 0.0


def run_qc_flags(row):
    """Registered section-3/V2-9 QC gates from recorded diagnostics only."""
    flags = []
    acc = row["accept"].get("scored")
    if acc is not None and not (0.5 <= acc <= 0.95):
        flags.append(f"scored_accept={acc:.3f} outside [0.5,0.95]")
    qc = row["qc_counters"]
    sc = qc.get("scored", {})
    if sc.get("proposals"):
        dfrac = sc["divergent"] / sc["proposals"]
        if dfrac > 0.001:
            flags.append(f"scored divergence rate {dfrac:.2e} > 0.1%")
    wa = qc.get("warmup", {})
    if wa.get("ensemble_steps"):
        for key in ("t_clamp", "eps_clamp", "l_max"):
            frac = wa[key] / wa["ensemble_steps"]
            if frac > 0.05:
                flags.append(f"warmup {key} rate {frac:.3f} > 5%")
    for ph in ("equil", "scored"):
        if qc.get(ph, {}).get("l_max", 0) > 0:
            flags.append(f"{ph} L_max event (code-defect tripwire, V2-9)")
    if row.get("freeze_clamp", 0):
        flags.append(f"freeze_clamp={row['freeze_clamp']}")
    for field in ("trace_var", "mse", "gold_mse"):
        val = row.get(field)
        if val is not None and not np.isfinite(val):
            flags.append(f"nonfinite {field} -> DIVERGED")
    return flags


def iqr_log2(vals):
    arr = np.log2(np.asarray(vals, dtype=np.float64))
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def qc_only_report(rows, lock_path, sidecar_dir, audit_dir):
    """V2-10 replicate-0 diagnostics-only mode.  Suppressed (not computed, not
    printed): ALL cross-arm quantities -- trace-variance ratios, A_T/A_eps and
    any schedule-agreement verdict, gold-MSE comparisons, probe z values,
    factorial contrasts, cross-fit shadow deltas."""
    manifest, _gold, _gsha, diverged = validate_campaign(
        rows, lock_path, sidecar_dir, audit_dir, full=False)
    print("== chees --qc-only (blinded: per-run diagnostics only; V2-10) ==")
    print(f"rows={len(rows)}; lock manifest {lock_path} OK (files+rows+sidecars"
          f"+S-7 artifacts match); seed/stream map OK; S-10 OK; frozenness "
          f"asserted on all closed rows")
    if diverged:
        print(f"  DIVERGED flags (sampler outcomes, V2-9; never rerun): {diverged}")
    trigger_cells = []
    for r in sorted(rows, key=lambda r: (r["dataset"], r["prior"], r["replicate"],
                                         r["kind"], r["arm"])):
        flags = run_qc_flags(r)
        lab = (f"{r['dataset']}/p={r['prior']}/rep{r['replicate']} "
               f"{r['kind']}:{r['arm']}[{r.get('variant', 'primary')}]")
        acc = {ph: round(v, 3) for ph, v in r["accept"].items()}
        qc = r["qc_counters"]
        div = {ph: qc[ph]["divergent"] for ph in qc}
        clamps = {ph: (qc[ph].get("t_clamp", 0), qc[ph].get("eps_clamp", 0),
                       qc[ph]["l_max"]) for ph in qc}
        azw = {ph: qc[ph].get("all_zero_weight", 0) for ph in qc}
        line = (f"  {lab}: accept={acc} divergent={div} clamps(T,eps,Lmax)={clamps} "
                f"all_zero_weight={azw}")
        if r["kind"] == "closed_loop":
            it = iqr_log2(r["T_star"])
            ie = iqr_log2(r["eps_star"])
            line += (f" warmup_end_abar={r['warmup_end_abar_mean']:.3f} "
                     f"freeze_clamp={r['freeze_clamp']} "
                     f"IQR_e(log2 T*)={it:.3f} IQR_e(log2 eps*)={ie:.3f}")
            if r["arm"] == "A" and r.get("variant", "primary") == "primary" and it > 1.0:
                trigger_cells.append((r["dataset"], r["prior"], r["replicate"], it))
        line += (" FLAGS: " + "; ".join(flags)) if flags else " qc-clean"
        print(line)
    if trigger_cells:
        print("  ARM-A INSTABILITY replicate flags (trigger evaluates on "
              "median over replicates when all 3 present):")
        for t in trigger_cells:
            print(f"    {t[0]}/p={t[1]} rep{t[2]}: IQR_e(log2 T*_A)={t[3]:.3f} > 1.0")
    print("  NOTE: cross-arm quantities suppressed per V2-10 (not computed).")


def ladder_reading(k1, coupled_unstable, qc_fail, h1a, h1b, h1c, cost_ok):
    """V3-5 first-match ladder with the V4-B rung (rung 0 K2/K3 is procedural,
    outside row data)."""
    if k1:
        return "UNRESOLVED (K1: arm-A M=8 iid instability; G=4 context only)"
    if coupled_unstable:
        return "FAILURE (coupled-arm instability sub-reading)"
    if qc_fail:
        return "FAILURE (QC/sampler-failure sub-reading; V4-B)"
    if not h1a and not h1b:
        return "FAILURE"
    if not h1c:
        return "P3 (bias gate failed; caps reading at PARTIAL)"
    if h1a and h1b:
        return ("DEPLOYMENT SUCCESS" if cost_ok
                else "VARIANCE-PER-ITERATION SUCCESS (cost gate failed)")
    return ("P1 (band without agreement)" if h1b
            else "P2 (agreement without band)")


def paired_probe_z(m_a, m_b):
    """V3-1 paired probe construction (ddof=1; degenerate guard)."""
    d = np.asarray(m_b, dtype=np.float64) - np.asarray(m_a, dtype=np.float64)
    dbar = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return (0.0, True) if dbar == 0.0 else (float("inf"), False)
    return dbar / (sd / math.sqrt(d.size)), True


def make_table(table_inputs, table_out, qc_only=False,
               lock_path=LOCK_MANIFEST_DEFAULT,
               sidecar_dir=Path("outputs/sweeps/chees_sidecars"),
               audit_dir=Path("outputs/selftests")):
    rows = load_rows(table_inputs)
    if qc_only:
        qc_only_report(rows, lock_path, sidecar_dir, audit_dir)
        return
    manifest, gold_map, _gsha, diverged_flags = validate_campaign(
        rows, lock_path, sidecar_dir, audit_dir, full=True)
    prim = [r for r in rows if r.get("variant", "primary") == "primary"]
    g4_rows = [r for r in rows if r.get("variant") == "g4"]
    cells = [(ds, prior) for ds in DATASETS for prior in PRIORS]

    def get(ds, prior, rep, kind, arm):
        for r in prim:
            if (r["dataset"], r["prior"], r["replicate"], r["kind"], r["arm"]) == \
                    (ds, prior, rep, kind, arm):
                return r
        raise KeyError((ds, prior, rep, kind, arm))

    stats = {}
    qc_fail_all = []
    probe_tripwires = []
    crossfit_report = []          # audit finding 7: emitted after unblinding
    # Closing-check residual 1: SEPARATE divergence sets -- only PRIMARY
    # divergence may touch H1/K1/QC ladder state; G4 divergence is contextual.
    diverged_cells = set()
    g4_diverged_cells = set()
    for ds_, prior_, rep_, variant_, reason in diverged_flags:
        if variant_ == "g4":
            g4_diverged_cells.add((ds_, prior_))
        else:
            diverged_cells.add((ds_, prior_))
            qc_fail_all.append(f"DIVERGED {ds_}/{prior_}/rep{rep_}: {reason}")
    for cell in cells:
        ds, prior = cell
        st = {"r_raw": [], "c_over_a": [], "rs_a": [], "rs_b": [], "m_src": [],
              "int_": [], "a_t": [], "a_eps": [], "gold_ratio": [], "probe_z": [],
              "probe_ok": [], "probe_z_ca": [], "probe_z_freeze": [],
              "r_sg": [], "r_tg": [], "r_bp": [], "r_wc": [],
              "iqr_T": {"A": [], "B": [], "C": []},
              "iqr_eps": {"A": [], "B": [], "C": []},
              "mean_l_scored": {"A": [], "B": [], "C": []},
              "tlen_b": [], "azw": 0, "tbar_a": [], "flags": []}
        for rep in range(3):
            ra = get(ds, prior, rep, "closed_loop", "A")
            rb = get(ds, prior, rep, "closed_loop", "B")
            rc = get(ds, prior, rep, "closed_loop", "C")
            rref = get(ds, prior, rep, "reference", "reference")
            fr = {(s, c): get(ds, prior, rep, "replay", f"replay_{s}_{c}")
                  for s, c in REPLAY_RUNS}
            # full QC applied to closed arms, the factorial, AND the auxiliary
            # reference (audit rec d)
            for r in [ra, rb, rc, rref, *fr.values()]:
                fl = run_qc_flags(r)
                if fl:
                    st["flags"].append(f"rep{rep}/{r['arm']}: " + "; ".join(fl))
                    qc_fail_all.extend(fl)
            for r in (ra, rb, rc):
                st["azw"] += sum(ph.get("all_zero_weight", 0)
                                 for ph in r["qc_counters"].values())
            # Fix-verification defect 3: cells already DIVERGED (validator
            # array flags or an earlier replicate) skip ALL cross-fit, ratio,
            # log, and probe arithmetic immediately -- diagnostics only.
            if cell in diverged_cells:
                continue
            for r in (ra, rb, rc):
                dcf = (np.log(np.asarray(r["T_star_cf"], dtype=np.float64))
                       - np.log(np.asarray(r["T_star"], dtype=np.float64)))
                crossfit_report.append(
                    (ds, prior, rep, r["arm"], float(np.median(dcf)),
                     float(np.percentile(dcf, 75) - np.percentile(dcf, 25)),
                     r["tuner_traj"]))
            # totality (audit finding 7): any nonpositive/nonfinite input
            # renders the cell DIVERGED; no ratio/log is ever attempted on it
            tvs = [ra["trace_var"], rb["trace_var"], rc["trace_var"]] + \
                  [fr[k]["trace_var"] for k in fr]
            gms = [ra["gold_mse"], rb["gold_mse"]]
            if not (all(np.isfinite(v) and v > 0.0 for v in tvs)
                    and all(np.isfinite(v) and v > 0.0 for v in gms)):
                diverged_cells.add(cell)
                qc_fail_all.append(f"DIVERGED {ds}/{prior}/rep{rep}: nonpositive "
                                   f"or nonfinite trace_var/gold_mse")
                continue
            st["r_raw"].append(rb["trace_var"] / ra["trace_var"])
            st["c_over_a"].append(rc["trace_var"] / ra["trace_var"])
            st["rs_a"].append(fr[("A", "zero_sum")]["trace_var"]
                              / fr[("A", "iid")]["trace_var"])
            st["rs_b"].append(fr[("B", "zero_sum")]["trace_var"]
                              / fr[("B", "iid")]["trace_var"])
            db_iid = math.log(fr[("B", "iid")]["trace_var"]) - \
                math.log(fr[("A", "iid")]["trace_var"])
            db_zs = math.log(fr[("B", "zero_sum")]["trace_var"]) - \
                math.log(fr[("A", "zero_sum")]["trace_var"])
            st["m_src"].append(0.5 * (db_iid + db_zs))
            st["int_"].append(db_zs - db_iid)
            lt = np.log(np.asarray(rb["T_star"])) - np.log(np.asarray(ra["T_star"]))
            le = np.log(np.asarray(rb["eps_star"])) - np.log(np.asarray(ra["eps_star"]))
            st["a_t"].append(float(np.exp(np.median(lt))))
            st["a_eps"].append(float(np.exp(np.median(le))))
            st["tbar_a"].append(float(np.median(ra["T_star"])))
            st["tlen_b"].append(float(np.median(rb["mean_realized_tlen"])))
            st["gold_ratio"].append(rb["gold_mse"] / ra["gold_mse"])
            z, ok = paired_probe_z(ra["m_probe"], rb["m_probe"])
            if not ok:
                probe_tripwires.append(f"{ds}/{prior}/rep{rep} B-vs-A scored probe")
            st["probe_z"].append(z)
            st["probe_ok"].append(abs(z) < 4.0)
            zc, okc = paired_probe_z(ra["m_probe"], rc["m_probe"])
            if not okc:
                probe_tripwires.append(f"{ds}/{prior}/rep{rep} C-vs-A scored probe")
            st["probe_z_ca"].append(zc)
            if ra.get("m_probe_freeze") and rb.get("m_probe_freeze"):
                zf, okf = paired_probe_z(ra["m_probe_freeze"],
                                         rb["m_probe_freeze"])
                if not okf:
                    probe_tripwires.append(f"{ds}/{prior}/rep{rep} freeze probe")
                st["probe_z_freeze"].append(zf)
            ug_a = ra["ug"]
            ug_b = rb["ug"]
            bp_a, bp_b, wc_a, wc_b = ra["bp"], rb["bp"], ra["wc"], rb["wc"]
            rr = st["r_raw"][-1]
            st["r_sg"].append(rr * (ug_b["scored"] / ug_a["scored"]))
            st["r_tg"].append(rr * (sum(ug_b.values()) / sum(ug_a.values())))
            st["r_bp"].append(rr * (sum(bp_b.values()) / sum(bp_a.values())))
            st["r_wc"].append(rr * (sum(wc_b.values()) / sum(wc_a.values())))
            for arm, r in (("A", ra), ("B", rb), ("C", rc)):
                st["iqr_T"][arm].append(iqr_log2(r["T_star"]))
                st["iqr_eps"][arm].append(iqr_log2(r["eps_star"]))
                st["mean_l_scored"][arm].append(r["mean_l"]["scored"])
        stats[cell] = st

    if probe_tripwires:
        raise SystemExit(
            "V2-9 DEFECT TRIPWIRE (audit finding 7): degenerate paired probe "
            "(zero SD, nonzero mean) at: " + "; ".join(probe_tripwires)
            + " -- bitwise-identical projections across arms are arithmetically "
            "implausible; halt, audit, and open the V2-9 defect procedure "
            "(archive rows, dated amendment, +600,000-block streams, rerun)")
    ok_cells = [c for c in cells if c not in diverged_cells]
    med = lambda v: float(np.median(v)) if v else float("nan")
    # H1a (V2-3): matched-pair joint gate + arm-B instability auto-fail
    at_med = {c: med(stats[c]["a_t"]) for c in cells}
    aeps_med = {c: med(stats[c]["a_eps"]) for c in cells}
    joint_band = [c for c in ok_cells
                  if 0.8 <= at_med[c] <= 1.25 and 0.8 <= aeps_med[c] <= 1.25]
    med_log_at = med([abs(math.log(at_med[c])) for c in ok_cells])
    med_log_aeps = med([abs(math.log(aeps_med[c])) for c in ok_cells])
    unstable_b = [c for c in ok_cells if med(stats[c]["iqr_T"]["B"]) > 1.0]
    h1a = (len(joint_band) >= 5 and med_log_at <= math.log(1.15)
           and med_log_aeps <= math.log(1.15) and not unstable_b
           and not diverged_cells)
    # H1b (V2-3, renamed): retention of the 0.81 ceiling
    rmed = {c: med(stats[c]["r_raw"]) for c in cells}
    h1b = (not diverged_cells and max(rmed[c] for c in ok_cells) <= 0.81
           and sum(rmed[c] <= 0.80 for c in ok_cells) >= 5)
    # H1c (V2-4): gold-MSE gate + scored-window probe gate in every cell
    gmed = {c: med(stats[c]["gold_ratio"]) for c in cells}
    probe_all = {c: (len(stats[c]["probe_ok"]) == 3 and all(stats[c]["probe_ok"]))
                 for c in cells}
    h1c = (not diverged_cells
           and sum(gmed[c] <= 1.25 for c in ok_cells) >= 5
           and all(gmed[c] <= 2.0 for c in ok_cells)
           and all(probe_all[c] for c in ok_cells))
    # V2-6 cost gate (deployment)
    rtg_med = {c: med(stats[c]["r_tg"]) for c in cells}
    cost_ok = (not diverged_cells
               and sum((rmed[c] <= 0.80 and rtg_med[c] < 1.0) for c in ok_cells) >= 5
               and max(rmed[c] for c in ok_cells) <= 0.81)
    # stability trigger (K1) on arm A
    trigger_a = [c for c in ok_cells if med(stats[c]["iqr_T"]["A"]) > 1.0]
    k1 = bool(trigger_a)
    qc_fail = bool(qc_fail_all) or bool(diverged_cells)
    reading = ladder_reading(k1, bool(unstable_b), qc_fail, h1a, h1b, h1c, cost_ok)

    labels = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}
    lines = ["\\begin{tabular}{llrrrrr}", "\\toprule",
             "Dataset & prior & closed $C_\\star$/iid & closed pairwise/iid & "
             "RS(A) & RS(B) & $A_T$ \\\\", "\\midrule"]
    for ds, prior in cells:
        st = stats[(ds, prior)]
        if (ds, prior) in diverged_cells:
            lines.append(f"{labels[ds]} & {prior} & \\multicolumn{{5}}{{c}}{{DIVERGED}} \\\\")
            continue
        lines.append(f"{labels[ds]} & {prior} & {nrl.cell(st['r_raw'])} & "
                     f"{nrl.cell(st['c_over_a'])} & {nrl.cell(st['rs_a'])} & "
                     f"{nrl.cell(st['rs_b'])} & {nrl.cell(st['a_t'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    cap = ["% CAPTION-REQUIRED (V2-5/V2-N/V3-5): freeze step 450 (EMA/dual-avg",
           "% freeze), tuner one-liner (ChEES Adam lr 0.025 beta2 0.95 on log T;",
           "% dual averaging to harmonic-mean acceptance 0.651; rotated-Halton",
           "% jitter), iid-arm across-replicate spread as noise floor, any QC",
           "% flags, and the three-replicate fragility caveat: per-cell medians",
           "% adjacent to the 0.80/0.81 thresholds are fragile; no verdict",
           "% stability under replicate deletion is claimed."]
    Path(table_out).parent.mkdir(parents=True, exist_ok=True)
    Path(table_out).write_text("\n".join(cap + lines), encoding="utf-8")

    out = ["== chees_closedloop.tex summary block (registered analysis, section 7) =="]
    for c in cells:
        ds, prior = c
        st = stats[c]
        if c in diverged_cells:
            out.append(f"  {ds}/p={prior}: DIVERGED (fails every gate it enters)")
            continue
        spread_a = (max(st["tbar_a"]) / min(st["tbar_a"])
                    if min(st["tbar_a"]) > 0 else float("nan"))
        out.append(
            f"  {ds}/p={prior}: B/A={med(st['r_raw']):.3f} C/A={med(st['c_over_a']):.3f} "
            f"RS(A)={med(st['rs_a']):.3f} RS(B)={med(st['rs_b']):.3f} "
            f"M_src={med(st['m_src']):.3f} INT={med(st['int_']):.3f} "
            f"A_T={at_med[c]:.3f} A_eps={aeps_med[c]:.3f} "
            f"gold MSE_B/MSE_A={gmed[c]:.3f} probe_z={[round(z, 2) for z in st['probe_z']]} "
            f"probe_z_CA={[round(z, 2) for z in st['probe_z_ca']]} "
            f"freeze_probe_z(S-4 analogue, non-gating"
            f"{', FLAG' if any(abs(z) >= 4 for z in st['probe_z_freeze']) else ''})="
            f"{[round(z, 2) for z in st['probe_z_freeze']]} "
            f"iidTbar spread={spread_a:.3f}")
        out.append(
            f"    cost: r_raw={med(st['r_raw']):.3f} r_sg={med(st['r_sg']):.3f} "
            f"r_tg={med(st['r_tg']):.3f} r_bp={med(st['r_bp']):.3f} "
            f"r_wc={med(st['r_wc']):.3f}; IQR_T A/B/C="
            f"{med(st['iqr_T']['A']):.3f}/{med(st['iqr_T']['B']):.3f}/"
            f"{med(st['iqr_T']['C']):.3f}; scored mean L A/B="
            f"{med(st['mean_l_scored']['A']):.2f}/{med(st['mean_l_scored']['B']):.2f}; "
            f"all_zero_weight={st['azw']}")
        cf_cell = [c4 for c4 in crossfit_report if (c4[0], c4[1]) == c]
        for arm in ("A", "B", "C"):
            vals = [(c4[2], c4[4], c4[5]) for c4 in cf_cell if c4[3] == arm]
            if vals:
                out.append(
                    f"    cross-fit shadow {arm} (V2-2): med_e[logT*cf-logT*] "
                    + " ".join(f"rep{v[0]}={v[1]:+.4f}(IQR {v[2]:.4f})"
                               for v in sorted(vals)))
        if st["flags"]:
            out.append(f"    QC FLAGS: {st['flags']}")
    out.append(f"  H1a: joint-band cells={len(joint_band)}/6 (need >=5), "
               f"median|log A_T|={med_log_at:.4f} (<= {math.log(1.15):.4f}), "
               f"median|log A_eps|={med_log_aeps:.4f}, arm-B unstable cells="
               f"{unstable_b} -> {'HOLDS' if h1a else 'FAILS'}")
    out.append(f"  H1b (0.81-ceiling retention): max={max((rmed[c] for c in ok_cells), default=float('nan')):.3f} "
               f"(<=0.81), n<=0.80={sum(rmed[c] <= 0.80 for c in ok_cells)}/6 "
               f"-> {'HOLDS' if h1b else 'FAILS'}")
    out.append(f"  H1c (gold bias + probe): n(med<=1.25)="
               f"{sum(gmed[c] <= 1.25 for c in ok_cells)}/6, max med="
               f"{max((gmed[c] for c in ok_cells), default=float('nan')):.3f} (<=2), "
               f"probe all-cells={'OK' if all(probe_all[c] for c in ok_cells) else 'FAIL'} "
               f"-> {'HOLDS' if h1c else 'FAILS'}")
    out.append(f"  cost gate (V2-6): {'PASS' if cost_ok else 'FAIL'}; "
               f"K1 trigger cells={trigger_a}; QC fail={bool(qc_fail)}")
    # V3-2 attribution registry (per breaching cell) / count-only rule
    breaches = [c for c in ok_cells if rmed[c] > 0.81]
    for c in breaches:
        ra_p = med(stats[c]["rs_a"]) <= 0.81
        rb_p = med(stats[c]["rs_b"]) <= 0.81
        lab = {(True, True): "UNIDENTIFIED CLOSED-LOOP HISTORY/FEEDBACK RESIDUAL",
               (True, False): "B-LEARNED SCHEDULE CONTENT implicated",
               (False, True): "OPERATING-POINT-DEPENDENT EXOGENOUS LOSS at the "
                              "A-learned schedule",
               (False, False): "EXOGENOUS MECHANISM LOSS at both learned "
                               "operating points"}[(ra_p, rb_p)]
        out.append(f"  attribution {c[0]}/p={c[1]}: RS(A) "
                   f"{'pass' if ra_p else 'fail'}, RS(B) "
                   f"{'pass' if rb_p else 'fail'} -> {lab}")
    if not breaches and sum(rmed[c] <= 0.80 for c in ok_cells) < 5 and not diverged_cells:
        k_low = sum(rmed[c] <= 0.80 for c in ok_cells)
        between = [c for c in ok_cells if 0.80 < rmed[c] <= 0.81]
        out.append(f"  count-only H1b failure: ceiling retained in all six cells; "
                   f"breadth criterion missed ({k_low} of 6 cells at or below "
                   f"0.80); cells in (0.80,0.81]: {between}; six-cell medians "
                   f"RS(A)={med([med(stats[c]['rs_a']) for c in ok_cells]):.3f} "
                   f"RS(B)={med([med(stats[c]['rs_b']) for c in ok_cells]):.3f} "
                   f"M_src={med([med(stats[c]['m_src']) for c in ok_cells]):.3f} "
                   f"INT={med([med(stats[c]['int_']) for c in ok_cells]):.3f} "
                   f"(non-categorical context)")
    # V2-N resonance diagnostic (descriptive only)
    if ("spambase", 0.3) in ok_cells:
        st = stats[("spambase", 0.3)]
        ml = med(st["mean_l_scored"]["B"])
        out.append(f"  resonance diagnostic spambase/0.3 (V2-N, descriptive; "
                   f"audit rec b): arm-B scored mean realized L={ml:.2f} vs "
                   f"[9,11]; median_e scored mean realized L*eps_e="
                   f"{med(st['tlen_b']):.2f} vs [14.4,17.6]")
    # G=4 secondary contextual report (V2-2/V3-5; audit finding 5): context
    # only, clustered n = reps/4; it never enters H1 verdicts.
    if g4_rows:
        out.append("  == G=4 SECONDARY VARIANT (context only; K1 verdict stays "
                   "UNRESOLVED; every number clustered, n = reps/4) ==")
        for c in cells:
            ds, prior = c
            sub = {(r["replicate"], r["arm"]): r for r in g4_rows
                   if (r["dataset"], r["prior"]) == c}
            if not sub:
                continue
            n_grp = CELL_REPS[ds] // G4_GROUP
            # Closing-check residual 2: cell-level skip BEFORE any ratio/IQR/
            # probe arithmetic -- one DIVERGED line, no partial-replicate stats.
            if (ds, prior) in g4_diverged_cells:
                out.append(f"    {ds}/p={prior} [clustered n={n_grp}]: DIVERGED "
                           f"(G4-only sampler divergence; context statistics "
                           f"suppressed; primary ladder unaffected)")
                continue
            ratios_ba, ratios_ca, iqr_a, iqr_b, zs = [], [], [], [], []
            for rep in range(3):
                ga, gb, gc = sub.get((rep, "A")), sub.get((rep, "B")), sub.get((rep, "C"))
                if not (ga and gb and gc):
                    continue
                ratios_ba.append(gb["trace_var"] / ga["trace_var"])
                ratios_ca.append(gc["trace_var"] / ga["trace_var"])
                iqr_a.append(iqr_log2(ga["T_star"]))
                iqr_b.append(iqr_log2(gb["T_star"]))
                ma = np.asarray(ga["m_probe"]).reshape(n_grp, G4_GROUP).mean(axis=1)
                mb = np.asarray(gb["m_probe"]).reshape(n_grp, G4_GROUP).mean(axis=1)
                zg, okg = paired_probe_z(ma, mb)
                if not okg:
                    raise SystemExit(f"V2-9 DEFECT TRIPWIRE: degenerate G4 "
                                     f"group probe at {ds}/{prior}/rep{rep}")
                zs.append(zg)
            out.append(f"    {ds}/p={prior} [clustered n={n_grp}]: "
                       f"B/A={med(ratios_ba):.3f} C/A={med(ratios_ca):.3f} "
                       f"IQR_g(log2 T*) A={med(iqr_a):.3f} B={med(iqr_b):.3f} "
                       f"group probe z={[round(z, 2) for z in zs]}")
    out.append(f"  lock/selftest status: manifest {lock_path} verified "
               f"(files+rows+{'54 G4 rows+' if g4_rows else ''}sidecars+S-7 GPU "
               f"artifacts); frozenness asserted on all closed rows; S-10 PASS "
               f"embedded")
    out.append(f"  READING (V3-5+V4-B first-match ladder): {reading}")
    # Companion cross-fit diagnostics file (audit finding 7): complete per-
    # (cell, arm, replicate) medians/IQRs + the every-10th-step (ghat_cf-ghat)
    # quartile series recorded in the rows.
    cf_path = Path(str(table_out) + ".crossfit.txt")
    cf_lines = ["# Cross-fitted centering shadow diagnostics (V2-2); plug-in "
                "re-integration at fixed recorded dynamics, NOT a closed-loop "
                "counterfactual."]
    for (ds, prior, rep, arm, mmed, miqr, ttraj) in crossfit_report:
        cf_lines.append(f"{ds}/p={prior}/rep{rep}/arm{arm}: "
                        f"med_e[logT*cf-logT*]={mmed:+.6f} IQR_e={miqr:.6f}")
        if ttraj and ttraj.get("step"):
            cf_lines.append("  step dghat_q25 dghat_med dghat_q75")
            for i, sn in enumerate(ttraj["step"]):
                cf_lines.append(f"  {sn} {ttraj['dghat_q25'][i]:.6g} "
                                f"{ttraj['dghat_med'][i]:.6g} "
                                f"{ttraj['dghat_q75'][i]:.6g}")
    cf_path.write_text("\n".join(cf_lines) + "\n", encoding="utf-8")
    out.append(f"  -> wrote {table_out} and {cf_path}")
    print("\n".join(out))


# --------------------------------------------------------------------------
# Selftests (K3: fix-or-abandon; every gate must pass before any real run)
# --------------------------------------------------------------------------

def _gauss():
    return nrl.gaussian_potential(SELFTEST_LAMS)


def load_legacy():
    """V3-3 frozen legacy oracle: assert the pinned sha256 BEFORE import; take
    every pass-1 symbol from this module only."""
    sha = full_sha256(LEGACY_PATH)
    assert sha == LEGACY_PIN, f"S-7 frozen-legacy hash mismatch: {sha} != {LEGACY_PIN}"
    spec = importlib.util.spec_from_file_location("nuts_legacy_36d66d6a",
                                                  LEGACY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def selftest_trio():
    """Reused jitter kernel exactness trio (verbatim from the substrate
    selftest, seeds 1,240,020 / 1,240,030 / 1,240,031), run against the
    REFACTORED kernels."""
    dim = 8
    potential = _gauss()
    var_exact = 1.0 / SELFTEST_LAMS
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
    va = nrl.jitter_leapfrog(x0, g0, p0, eps_t, l_const, potential)
    vb = plain_leapfrog(x0, g0, p0, 7)
    worst_const = max(float(cp.max(cp.abs(a - b)).get()) for a, b in zip(va, vb))
    print(f"selftest jitter constant-L vs plain leapfrog: max abs diff {worst_const:.3e}")
    assert worst_const < 1e-6, "constant-L jitter mismatch"
    l_mixed = cp.asarray([3, 3, 7, 7, 12, 12, 20, 20], dtype=cp.int32)
    vm = nrl.jitter_leapfrog(x0, g0, p0, eps_t, l_mixed, potential)
    worst_mixed = 0.0
    for l_val in (3, 7, 12, 20):
        sel = cp.asarray(l_mixed.get() == l_val)
        vs = nrl.jitter_leapfrog(x0[sel], g0[sel], p0[sel], eps_t,
                                 cp.full(int(sel.sum()), l_val, dtype=cp.int32),
                                 potential)
        for a, b in zip((vm[0][sel], vm[1][sel], vm[2][sel], vm[3][sel]), vs):
            worst_mixed = max(worst_mixed, float(cp.max(cp.abs(a - b)).get()))
    print(f"selftest jitter mixed-L group equivalence: max abs diff {worst_mixed:.3e}")
    assert worst_mixed < 1e-6, "mixed-L masked continuation corrupts trajectories"
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
        lj = cp.asarray(rng_l.integers(1, 21, size=512), dtype=cp.int32)
        xj, uj, gj = nrl.jitter_hmc_step(xj, uj, gj, p0j, 0.25, lj, potential, acc)
        if step >= 300:
            m1 += xj.mean(axis=0)
            m2 += (xj * xj).mean(axis=0)
            kept += 1
    mean_j = (m1 / kept).get()
    second_j = (m2 / kept).get()
    zj_mean = np.abs(mean_j) / np.sqrt(var_exact / 512)
    zj_sec = np.abs(second_j - var_exact) / np.sqrt(2.0 * var_exact ** 2 / 512)
    print(f"selftest jitter MH moments: max|z_mean|={zj_mean.max():.2f} "
          f"max|z_second|={zj_sec.max():.2f} accept={float((acc / 1500).get()):.3f}")
    assert zj_mean.max() < 4.0 and zj_sec.max() < 4.0, "jitter MH kernel moments off"
    print("selftest trio PASS")


def selftest_s2():
    """S-2 (V2-8 fixtures, amendment v5): criterion direction at T=1.5 (+)
    and T=6 (-), 200 batches x 256 ensembles x 8 chains, exact-pi states,
    eps 0.25, rotated Halton with index = batch number, seed 1,250,001; then
    the full closed-loop freeze run (seed 1,250,002), median T* in [2.5, 7.0]."""
    dim, n_ens, k = 8, 256, K_CHAINS
    potential = _gauss()
    sd = cp.asarray((1.0 / np.sqrt(SELFTEST_LAMS)).astype(np.float32))
    eps = SELFTEST_EPS0

    def direction(t_value):
        u_rot = cp.asarray(np.random.default_rng(1250001).random(n_ens))
        cp.random.seed(1250001)
        batch_means = np.zeros(200)
        for b in range(1, 201):
            x = cp.random.standard_normal((n_ens * k, dim), dtype=cp.float32) * sd[None, :]
            u_pot, g = potential(x)
            p0 = cp.random.standard_normal((n_ens * k, dim), dtype=cp.float32)
            h = (halton2(b) + u_rot) % 1.0
            t_len = h * t_value
            lf = cp.ceil(t_len / eps)
            l_unit = cp.minimum(cp.maximum(lf, 1.0), float(L_MAX)).astype(cp.int32)
            l_row = cp.repeat(l_unit, k)
            xf, uf, gf, pf, la, alpha = nrl.jitter_propose(x, u_pot, g, p0, eps,
                                                           l_row, potential)
            ghat, _ = chees_gradient(x.reshape(n_ens, k, dim),
                                     xf.reshape(n_ens, k, dim),
                                     pf.reshape(n_ens, k, dim),
                                     alpha.reshape(n_ens, k), t_len)
            batch_means[b - 1] = float(ghat.mean())
        m = batch_means.mean()
        se = batch_means.std(ddof=1) / math.sqrt(200)
        return m, se

    # Amendment v5: probes are (T=1.5, positive) and (T=6, negative).  The
    # original T=9 probe was retired as unsatisfiable (second positive lobe;
    # dominant-mode gradient zeros solve tan a = a at T=4.4934/7.7253).
    failures = []
    for t_value, want in ((1.5, +1.0), (6.0, -1.0)):
        m, se = direction(t_value)
        ok = want * m / se >= 3.0
        print(f"S-2 direction T={t_value}: mean ghat={m:.4g} SE={se:.4g} "
              f"z={m / se:.2f} (registered v5 gate: sign "
              f"{'+' if want > 0 else '-'} at >=3 SE) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"direction T={t_value}")
    u_rot = np.random.default_rng(1250002).random(n_ens)
    res = run_closed_loop(potential, dim, n_ens, k, "iid", SELFTEST_EPS0, 1250002,
                          u_rot, crossfit=True, exact_center_shadow=True)
    med_t = float(np.median(res["T_star"]))
    med_cf = float(np.median(res["T_star_cf"]))
    med_ex = float(np.median(res["T_star_exact_center"]))
    print(f"S-2 closed-loop freeze: median T*={med_t:.3f} (gate [2.5, 7.0]); "
          f"shadow freezes: cross-fitted {med_cf:.3f}, exact-center {med_ex:.3f} "
          f"(V2-2 centering variants); scored accept={res['accept']['scored']:.3f}")
    if not (2.5 <= med_t <= 7.0):
        failures.append(f"freeze median T* {med_t:.3f} outside [2.5, 7.0]")
    assert not failures, ("S-2 FAILS (K3 fix-or-abandon; campaign blocked): "
                          + "; ".join(failures))
    print("S-2 PASS")


def selftest_s3a():
    """S-3a stationary-kernel gate (V2-1): schedules from two independent
    450-step warmups (iid 1,250,010 / zero_sum 1,250,011); four source x
    coupling scoring combinations (1,250,012-15), fresh exact-pi starts,
    one-step and 200-step snapshot moments, ensemble-clustered MCSE, |z|<4."""
    dim, n_ens, k = 8, 512, K_CHAINS
    potential = _gauss()
    var_exact = 1.0 / SELFTEST_LAMS
    sd_np = (1.0 / np.sqrt(SELFTEST_LAMS)).astype(np.float32)
    warm = {}
    for coup, seed in (("iid", 1250010), ("zero_sum", 1250011)):
        u_rot = np.random.default_rng(seed).random(n_ens)
        res = run_closed_loop(potential, dim, n_ens, k, coup, SELFTEST_EPS0, seed,
                              u_rot, crossfit=False, n_equil=0, n_score=0)
        warm[coup] = res
        print(f"S-3a warmup {coup}: median T*={np.median(res['T_star']):.3f} "
              f"median eps*={np.median(res['eps_star']):.4f} "
              f"freeze_clamp={res['freeze_clamp']}")
    worst = 0.0
    for si, src in enumerate(("iid", "zero_sum")):
        for ci, coup in enumerate(("iid", "zero_sum")):
            seed = 1250012 + si * 2 + ci
            cp.random.seed(seed)
            x0 = (cp.random.standard_normal((n_ens, k, dim), dtype=cp.float32)
                  * cp.asarray(sd_np)[None, None, :])
            u_rot = np.random.default_rng(seed).random(n_ens)
            res = run_frozen(potential, dim, n_ens, k, coup, u_rot,
                             np.asarray(warm[src]["T_star"], dtype=np.float64),
                             np.asarray(warm[src]["eps_star"], dtype=np.float64),
                             steps=200, burnin=200, x0=x0, snapshots=(1, 200))
            for horizon in (1, 200):
                x3 = res["snapshots"][horizon].astype(np.float64)
                unit1 = x3.mean(axis=1)
                unit2 = (x3 ** 2).mean(axis=1)
                z1 = np.abs(unit1.mean(axis=0)) / (unit1.std(axis=0, ddof=1)
                                                   / math.sqrt(n_ens))
                z2 = np.abs(unit2.mean(axis=0) - var_exact) / (
                    unit2.std(axis=0, ddof=1) / math.sqrt(n_ens))
                zmax = max(z1.max(), z2.max())
                worst = max(worst, zmax)
                print(f"S-3a {src}-learned x {coup} seed {seed} step {horizon}: "
                      f"max|z1|={z1.max():.2f} max|z2|={z2.max():.2f}")
                assert zmax < 4.0, f"S-3a gate fails: {src}/{coup}/step{horizon}"
    print(f"S-3a PASS (worst |z|={worst:.2f})")


def selftest_s3b():
    """S-3b continuation/transient test (V2-1; PASS = transient below detection
    at this registered resolution, nothing stronger)."""
    dim, n_ens, k = 8, 512, K_CHAINS
    potential = _gauss()
    var_exact = 1.0 / SELFTEST_LAMS
    se = np.sqrt(2.0 * var_exact ** 2 / n_ens)
    for coup, seed in (("iid", 1250020), ("zero_sum", 1250021)):
        u_rot = np.random.default_rng(seed).random(n_ens)
        res = run_closed_loop(potential, dim, n_ens, k, coup, SELFTEST_EPS0, seed,
                              u_rot, crossfit=False)
        m2 = res["m2_chain0_scored"]
        z = np.abs(m2 - var_exact) / se
        print(f"S-3b {coup}: max|z|={z.max():.2f} (scored-window chain-0 second "
              f"moments vs exact; single-snapshot SE convention)")
        assert z.max() < 4.0, f"S-3b gate fails under {coup}"
    print("S-3b PASS (transient below detection at registered resolution)")


def selftest_s4():
    """S-4 (DEMOTED, non-gating): freeze-step zero_sum-vs-iid chain-0 second
    moments (seeds 1,250,030/1,250,031); recorded, flagged if |z|>=4."""
    dim, n_ens, k = 8, 512, K_CHAINS
    potential = _gauss()
    var_exact = 1.0 / SELFTEST_LAMS
    snaps = {}
    for coup, seed in (("iid", 1250030), ("zero_sum", 1250031)):
        u_rot = np.random.default_rng(seed).random(n_ens)
        res = run_closed_loop(potential, dim, n_ens, k, coup, SELFTEST_EPS0, seed,
                              u_rot, crossfit=False, n_equil=0, n_score=0)
        snaps[coup] = res["freeze_chain0_sq"].mean(axis=0)
    se = np.sqrt(2.0 * var_exact ** 2 / n_ens)
    z = np.abs(snaps["zero_sum"] - snaps["iid"]) / (se * math.sqrt(2.0))
    flag = " FLAG(|z|>=4)" if z.max() >= 4.0 else ""
    print(f"S-4 (non-gating transient diagnostic): freeze-step zs-vs-iid "
          f"max|z|={z.max():.2f}{flag}")
    print("S-4 recorded (gates nothing)")


def selftest_s5():
    """S-5 freeze mechanics (seed 1,250,040): post-freeze (T, eps) bitwise
    constant (in-run asserts) and the scored-phase transition recomputed
    externally as one plain recomposed jitter step at (T*, eps*) -- the
    kernel's own exactness delegates to the reused trio."""
    dim, n_ens, k = 8, 64, K_CHAINS
    potential = _gauss()
    u_rot = np.random.default_rng(1250040).random(n_ens)
    cap = {"steps": {N_WARM + N_EQUIL, N_WARM + N_EQUIL + 1}, "data": {}}
    res = run_closed_loop(potential, dim, n_ens, k, "iid", SELFTEST_EPS0, 1250040,
                          u_rot, crossfit=False, capture=cap)
    n1 = N_WARM + N_EQUIL + 1
    d = cap["data"][n1]
    assert np.array_equal(d["T_use"], res["T_star"]), "S-5: scored T != T*"
    assert np.array_equal(d["eps_use"], res["eps_star"]), "S-5: scored eps != eps*"
    h = (halton2(n1) + d["u_rot_now"]) % 1.0
    assert np.array_equal(h, d["h"]), "S-5: jitter fraction mismatch"
    lf = np.ceil(h * res["T_star"] / res["eps_star"])
    l_ens = np.clip(lf, 1.0, float(L_MAX)).astype(np.int32)
    assert np.array_equal(np.repeat(l_ens, k), d["l_row"]), "S-5: L rule mismatch"
    x_pre = cp.asarray(d["x_pre"])
    u_pre, g_pre = potential(x_pre)
    eps_w = cp.repeat(cp.asarray(res["eps_star"]).astype(cp.float32), k)[:, None]
    acc = cp.zeros((), dtype=cp.float64)
    xf, uf, gf, pf, la, alpha = nrl.jitter_propose(
        x_pre, u_pre, g_pre, cp.asarray(d["p0"]), eps_w,
        cp.asarray(d["l_row"]), potential)
    (xn, un, gn), _acc, _uu = nrl.jitter_accept(
        x_pre, u_pre, g_pre, xf, uf, gf, la, acc,
        u_unif=cp.asarray(d["u_unif"]))
    assert np.array_equal(xn.get(), d["x_post"]), "S-5: recomposed step != loop step (x)"
    assert np.array_equal(un.get(), d["u_post"]), "S-5: recomposed step != loop step (u)"
    assert np.array_equal(gn.get(), d["g_post"]), "S-5: recomposed step != loop step (g)"
    print("S-5 PASS: post-freeze (T, eps) bitwise constant (in-run asserts) and "
          "scored step == plain recomposed jitter kernel at (T*, eps*) bitwise "
          "(kernel exactness delegated to the reused trio)")


def selftest_s6():
    """S-6 dual averaging (V3-4 fixtures): eps_0 = 0.50 = 2x the stable 0.25,
    mu/eps-box at eps_0 = 0.50, iid, 256x8, full 650 steps, seed 1,250,050;
    harmonic-mean scored acceptance aggregate in [0.55, 0.75]."""
    dim, n_ens, k = 8, 256, K_CHAINS
    potential = _gauss()
    u_rot = np.random.default_rng(1250050).random(n_ens)
    res = run_closed_loop(potential, dim, n_ens, k, "iid", 0.50, 1250050, u_rot,
                          crossfit=False)
    agg = res["scored_abar_mean"]
    print(f"S-6 dual averaging from eps_0=0.50: scored harmonic-acceptance "
          f"aggregate={agg:.3f} (gate [0.55, 0.75]); median eps*="
          f"{np.median(res['eps_star']):.4f}; warmup-end abar="
          f"{res['warmup_end_abar_mean']:.3f}")
    assert 0.55 <= agg <= 0.75, f"S-6 gate fails: {agg}"
    print("S-6 PASS")


def _sha_arrays(*arrays):
    hh = hashlib.sha256()
    for a in arrays:
        hh.update(np.ascontiguousarray(a).tobytes())
    return hh.hexdigest()


def _check_dh_finite(u_pre, p0, uf, pf, label):
    """V4-A contract, HARD (audit finding 3): H0, H, and H0-H must be finite
    on EVERY compared step -- NaN, -inf, +inf, or an individually nonfinite
    energy is an S-7 FAILURE (the +inf case additionally being the
    legacy/totalized divergence tripwire), never a note."""
    h0 = (u_pre.astype(cp.float64) + 0.5 * nrl.dot64(p0, p0)).get()
    hn = (uf.astype(cp.float64) + 0.5 * nrl.dot64(pf, pf)).get()
    assert np.all(np.isfinite(h0)), f"S-7 FAIL at {label}: nonfinite H0 (V4-A)"
    assert np.all(np.isfinite(hn)), f"S-7 FAIL at {label}: nonfinite H (V4-A)"
    assert np.all(np.isfinite(h0 - hn)), (
        f"S-7 FAIL at {label}: nonfinite H0-H (V4-A; +inf = the "
        f"legacy/totalized divergence tripwire -- HALT AND AUDIT)")


def selftest_s7(data_dir=Path("data/real"), audit_dir=Path("outputs/selftests")):
    """S-7 legacy regression (V2-5/V3-3/V4-A): pass 1 = frozen legacy module
    ONLY; pass 2 = recomposed propose+accept; bitwise stepwise equality of
    (x, u, g) and the acceptance accumulator; p0/uniform/L rows archived."""
    leg = load_legacy()
    audit_dir.mkdir(parents=True, exist_ok=True)
    suffix = "cpustub" if IS_STUB else "gpu"

    def two_pass(tag, pot_leg, pot_new, make_p0_leg, make_p0_new, x_init, eps,
                 l_rows, dyn_seed, n_walk, score_from=None, reps_k=None):
        """Runs pass1 (legacy), RNG uniform replay, pass2 (recomposed); returns
        (per-step sha list, acc1, acc2, uniforms, trace_var or None)."""
        # pass 1: frozen legacy kernel, natural draw order
        cp.random.seed(dyn_seed)
        x = x_init()
        u, g = pot_leg(x)
        acc1 = cp.zeros((), dtype=cp.float32)
        shas, p0_shas = [], []
        p0_arch = np.empty((len(l_rows), n_walk, x.shape[1]), dtype=np.float32)
        ens_acc = None
        if score_from is not None:
            ens_acc = cp.zeros((reps_k[0], x.shape[1]), dtype=cp.float32)
        for i, l_row in enumerate(l_rows):
            p0 = make_p0_leg()
            x, u, g = leg.jitter_hmc_step(x, u, g, p0, eps,
                                          cp.asarray(l_row, dtype=cp.int32),
                                          pot_leg, acc1)
            shas.append(_sha_arrays(x.get(), u.get(), g.get()))
            p0_arch[i] = p0.get()
            p0_shas.append(_sha_arrays(p0_arch[i]))
            if score_from is not None and i >= score_from:
                ens_acc += x.reshape(reps_k[0], reps_k[1], -1).mean(axis=1)
        tv = None
        if score_from is not None:
            ens = ens_acc / (len(l_rows) - score_from)
            cen = ens - ens.mean(axis=0)[None, :]
            tv = float(cp.mean(cp.sum(cen * cen, axis=1)).get())
        # RNG replay: recompute the acceptance uniforms from the shared stream
        cp.random.seed(dyn_seed)
        unifs = np.empty((len(l_rows), n_walk), dtype=np.float32)
        for i in range(len(l_rows)):
            p0r = make_p0_leg()
            assert _sha_arrays(p0r.get()) == p0_shas[i], "S-7 RNG replay p0 drift"
            unifs[i] = cp.random.random((n_walk,), dtype=cp.float32).get()
        # pass 2: recomposed propose + accept (refactored module), identical
        # contractual call order (p0 outside, uniform inside accept)
        cp.random.seed(dyn_seed)
        x = x_init()
        u, g = pot_new(x)
        acc2 = cp.zeros((), dtype=cp.float32)
        for i, l_row in enumerate(l_rows):
            p0 = make_p0_new()
            u_pre = u
            xf, uf, gf, pf, la, alpha = nrl.jitter_propose(
                x, u, g, p0, eps, cp.asarray(l_row, dtype=cp.int32), pot_new)
            _check_dh_finite(u_pre, p0, uf, pf, f"{tag} step {i}")  # hard (V4-A)
            (x, u, g), _a, uu = nrl.jitter_accept(x, u, g, xf, uf, gf, la, acc2)
            assert np.array_equal(uu.get(), unifs[i]), f"S-7 {tag} uniform stream drift step {i}"
            sha2 = _sha_arrays(x.get(), u.get(), g.get())
            assert sha2 == shas[i], f"S-7 {tag}: pass-2 diverges from legacy at step {i}"
        assert float(acc1.get()) == float(acc2.get()), f"S-7 {tag}: acceptance accumulator differs"
        return shas, float(acc1.get()) / len(l_rows), unifs, tv, p0_arch

    # Config (i): Gaussian, 512 walkers, eps 0.25, 200 steps, seed 1,250,060
    rng = np.random.default_rng(1250060)
    l_fix = rng.integers(1, 21, size=(200, 512))
    got = hashlib.sha256(l_fix.astype("<i8").tobytes(order="C")).hexdigest()
    assert got == S7_FIX_GAUSS_SHA, f"S-7 gaussian L-fixture hash {got}"
    pot_leg = leg.gaussian_potential(SELFTEST_LAMS)
    pot_new = _gauss()
    # AUDIT NOTE (finding 3): the three S-7 audit artifacts written below are
    # only valid as a SET when regenerated in ONE final GPU --selftest run
    # under ONE triple hash (the executable-hash-lock bytes); mixed-hash or
    # CPU-stub artifacts never satisfy the lock validator.
    shas, arate, unifs, _, p0_arch = two_pass(
        "gaussian", pot_leg, pot_new,
        lambda: cp.random.standard_normal((512, 8), dtype=cp.float32),
        lambda: cp.random.standard_normal((512, 8), dtype=cp.float32),
        lambda: cp.zeros((512, 8), dtype=cp.float32), 0.25,
        list(l_fix), 1250060, 512)
    np.savez_compressed(audit_dir / f"chees_s7_gaussian_{suffix}.npz",
                        l_rows=l_fix.astype(np.int32), uniforms=unifs,
                        p0=p0_arch, state_shas=np.asarray(shas),
                        accept_rate=arate,
                        provenance=json.dumps(provenance(), sort_keys=True))
    print(f"S-7 gaussian: 200 steps bitwise stepwise equality PASS "
          f"(accept={arate:.3f}); audit -> chees_s7_gaussian_{suffix}.npz")

    # Config (ii): archived spambase/0.3 replicate-0 jitter_shared
    rng = np.random.default_rng(S7_L_SEED)
    full = rng.integers(1, 21, size=(900, 96))
    assert hashlib.sha256(full.astype("<i8").tobytes(order="C")).hexdigest() \
        == S7_FIX_FULL_SHA, "S-7 spambase full L-fixture hash"
    sub = full[600:900]
    assert hashlib.sha256(sub.astype("<i8").tobytes(order="C")).hexdigest() \
        == S7_FIX_SUB_SHA, "S-7 spambase consumed L-fixture hash"
    dx_leg, dy_leg = leg.load_dataset(data_dir, "spambase")
    dx_new, dy_new = nrl.load_dataset(data_dir, "spambase")
    dim = int(dx_leg.shape[1])
    eps_p, rate_p = leg.calibrate_eps("spambase", dim, 0.3, dx_leg, dy_leg, 1024)
    if eps_p != S7_EPS_ARCHIVED:
        msg = (f"S-7 pilot eps {eps_p} != archived {S7_EPS_ARCHIVED} "
               f"(pilot rate {rate_p:.3f})")
        if IS_STUB:
            print(f"  SMOKE-WARN: {msg} -- stub RNG differs from cuRAND; the "
                  f"registered exact-equality gate is GPU-DEFERRED; proceeding "
                  f"at archived eps 1.6")
        else:
            raise AssertionError(msg)
    pot_leg = leg.logistic_potential(dx_leg, dy_leg, 0.3, 1024)
    pot_new = nrl.logistic_potential(dx_new, dy_new, 0.3, 1024)
    l_rows = [np.repeat(sub[i], K_CHAINS) for i in range(300)]
    for coup in ("iid", "zero_sum"):
        shas, arate, unifs, tv, p0_arch = two_pass(
            f"spambase/{coup}", pot_leg, pot_new,
            lambda c=coup: leg.make_noise((96, K_CHAINS, dim), c).reshape(-1, dim),
            lambda c=coup: nrl.make_noise((96, K_CHAINS, dim), c).reshape(-1, dim),
            lambda: cp.zeros((96 * K_CHAINS, dim), dtype=cp.float32),
            S7_EPS_ARCHIVED, l_rows, S7_DYN[coup], 96 * K_CHAINS,
            score_from=150, reps_k=(96, K_CHAINS))
        np.savez_compressed(audit_dir / f"chees_s7_spambase_{coup}_{suffix}.npz",
                            l_rows=sub.astype(np.int32), uniforms=unifs,
                            p0=p0_arch, state_shas=np.asarray(shas),
                            accept_rate=arate, trace_var_pass1=tv,
                            trace_var_archived=S7_TRACEVAR_ARCHIVED[coup],
                            provenance=json.dumps(provenance(), sort_keys=True))
        print(f"S-7 spambase/0.3 {coup}: 300 steps bitwise stepwise equality "
              f"PASS (accept={arate:.3f}); pass-1 trace_var={tv:.6g} vs archived "
              f"row {S7_TRACEVAR_ARCHIVED[coup]:.6g} (context only, no gate"
              f"{'; stub RNG differs from the archived cuRAND stream' if IS_STUB else ''})")
    print("S-7 PASS")


def selftest_s8():
    """S-8 mixed-(L, eps_e) group equivalence (V3-4 fixtures, seed 1,250,070):
    one batched propose vs four frozen-legacy scalar-eps reference paths."""
    leg = load_legacy()
    pot = _gauss()
    pot_leg = leg.gaussian_potential(SELFTEST_LAMS)
    cp.random.seed(1250070)
    x0 = 0.5 * cp.random.standard_normal((16, 8), dtype=cp.float32)
    p0 = cp.random.standard_normal((16, 8), dtype=cp.float32)
    eps_e = np.asarray([0.15, 0.2, 0.3, 0.45], dtype=np.float64)
    l_e = [3, 7, 12, 20]
    eps32 = eps_e.astype(np.float32)
    eps_w = cp.asarray(np.repeat(eps32, 4))[:, None]
    l_row = cp.asarray(np.repeat(l_e, 4), dtype=cp.int32)
    u0, g0 = pot(x0)
    xf, uf, gf, pf, la, alpha = nrl.jitter_propose(x0, u0, g0, p0, eps_w, l_row, pot)
    worst = 0.0
    for gi in range(4):
        sl = slice(4 * gi, 4 * gi + 4)
        _ug, gg = pot_leg(x0[sl])
        ref = leg.jitter_leapfrog(x0[sl], gg, p0[sl], eps32[gi],
                                  cp.full(4, l_e[gi], dtype=cp.int32), pot_leg)
        for a, b in zip((xf[sl], uf[sl], gf[sl], pf[sl]), ref):
            worst = max(worst, float(cp.max(cp.abs(a - b)).get()))
    print(f"S-8 mixed-(L, eps_e) vs frozen-legacy scalar paths: max abs diff "
          f"{worst:.3e} (< 1e-6)")
    assert worst < 1e-6, "S-8 mixed-(L,eps) group equivalence fails"
    print("S-8 PASS")
    return (x0, p0, u0, g0, eps32, l_e, (xf, uf, gf, pf, la, alpha))


def _alpha_recompute_ok(u_pre, p0, uf, pf, alpha):
    h0 = u_pre.astype(cp.float64) + 0.5 * nrl.dot64(p0, p0)
    hn = uf.astype(cp.float64) + 0.5 * nrl.dot64(pf, pf)
    dh = h0 - hn
    la2 = cp.where(cp.isfinite(dh), cp.minimum(dh, 0.0), -cp.inf)
    return np.array_equal(cp.exp(la2).get(), alpha.get())


def selftest_s9():
    """S-9 propose/accept contract (V3-4 fixtures): (i) returned-alpha bitwise
    equality; (ii) endpoint momentum vs frozen legacy; (iii-a) scalar-eps
    recomposition vs the frozen legacy kernel (seed 1,250,081); (iii-b)
    mixed-(L, eps) full-step group equivalence under injected fixtures."""
    leg = load_legacy()
    pot = _gauss()
    # (i) + (ii) on the S-8 batch
    s8 = selftest_s8()
    x0s, p0s, u0s, g0s, eps32s, l_es, out = s8
    xfs, ufs, gfs, pfs, las, alphas = out
    assert _alpha_recompute_ok(u0s, p0s, ufs, pfs, alphas), "S-9(i) fails on S-8 batch"
    worst_pf = 0.0
    pot_leg = leg.gaussian_potential(SELFTEST_LAMS)
    for gi in range(4):
        sl = slice(4 * gi, 4 * gi + 4)
        _ug, gg = pot_leg(x0s[sl])
        ref = leg.jitter_leapfrog(x0s[sl], gg, p0s[sl], eps32s[gi],
                                  cp.full(4, l_es[gi], dtype=cp.int32), pot_leg)
        worst_pf = max(worst_pf, float(cp.max(cp.abs(pfs[sl] - ref[3])).get()))
    assert worst_pf < 1e-6, "S-9(ii) endpoint-momentum equality fails"
    print(f"S-9(i) returned-alpha bitwise OK on S-8 batch; S-9(ii) endpoint "
          f"momentum max abs diff {worst_pf:.3e}")
    # Registered fixtures (numpy 1,250,080 schedule; cupy 1,250,080 dynamics)
    rng = np.random.default_rng(1250080)
    eps_e = 0.25 * 2.0 ** rng.uniform(-1.0, 1.0, size=64)      # float64 FIRST
    l_sched = rng.integers(1, 21, size=(50, 64))               # THEN L
    cp.random.seed(1250080)
    x0 = 0.5 * cp.random.standard_normal((512, 8), dtype=cp.float32)
    u_fix = cp.random.random((50, 512), dtype=cp.float32)
    p_fix = cp.random.standard_normal((50, 512, 8), dtype=cp.float32)
    # (iii-a) scalar-eps recomposition vs frozen legacy kernel, seed 1,250,081
    eps_sc = np.float32(0.25)
    states = []
    cp.random.seed(1250081)
    x, (u, g) = x0.copy(), pot_leg(x0)
    acc1 = cp.zeros((), dtype=cp.float32)
    for n in range(50):
        p0 = cp.random.standard_normal((512, 8), dtype=cp.float32)
        l_row = cp.asarray(np.repeat(l_sched[n], 8), dtype=cp.int32)
        x, u, g = leg.jitter_hmc_step(x, u, g, p0, eps_sc, l_row, pot_leg, acc1)
        states.append((x.get(), u.get(), g.get()))
    cp.random.seed(1250081)
    x, (u, g) = x0.copy(), pot(x0)
    acc2 = cp.zeros((), dtype=cp.float32)
    for n in range(50):
        p0 = cp.random.standard_normal((512, 8), dtype=cp.float32)
        l_row = cp.asarray(np.repeat(l_sched[n], 8), dtype=cp.int32)
        u_pre = u
        xf, uf, gf, pf, la, alpha = nrl.jitter_propose(x, u, g, p0, eps_sc,
                                                       l_row, pot)
        assert _alpha_recompute_ok(u_pre, p0, uf, pf, alpha), f"S-9(i) fails step {n}"
        (x, u, g), _a, _uu = nrl.jitter_accept(x, u, g, xf, uf, gf, la, acc2)
        for a, b in zip((x.get(), u.get(), g.get()), states[n]):
            assert np.array_equal(a, b), f"S-9(iii-a) bitwise mismatch step {n}"
    assert float(acc1.get()) == float(acc2.get()), "S-9(iii-a) acceptance accumulator"
    print("S-9(iii-a) scalar-eps recomposition vs frozen legacy: 50 steps bitwise OK")
    # (iii-b) mixed full step vs per-group scalar full step, injected fixtures
    eps32 = eps_e.astype(np.float32)
    eps_w = cp.asarray(np.repeat(eps32, 8))[:, None]
    xm, (um, gm) = x0.copy(), pot(x0)
    xg = [x0[8 * gi:8 * gi + 8].copy() for gi in range(64)]
    ug_ = [None] * 64
    gg_ = [None] * 64
    for gi in range(64):
        ug_[gi], gg_[gi] = pot(xg[gi])
    accm = cp.zeros((), dtype=cp.float64)
    accg = cp.zeros((), dtype=cp.float64)
    for n in range(50):
        l_units = l_sched[n]
        l_row = cp.asarray(np.repeat(l_units, 8), dtype=cp.int32)
        p0 = p_fix[n]
        uu = u_fix[n]
        u_pre = um
        xf, uf, gf, pf, la, alpha = nrl.jitter_propose(xm, um, gm, p0, eps_w,
                                                       l_row, pot)
        assert _alpha_recompute_ok(u_pre, p0, uf, pf, alpha), f"S-9(i) mixed step {n}"
        (xm, um, gm), _a, _u2 = nrl.jitter_accept(xm, um, gm, xf, uf, gf, la,
                                                  accm, u_unif=uu)
        for gi in range(64):
            sl = slice(8 * gi, 8 * gi + 8)
            xfg, ufg, gfg, pfg, lag, _alg = nrl.jitter_propose(
                xg[gi], ug_[gi], gg_[gi], p0[sl], eps32[gi],
                cp.full(8, int(l_units[gi]), dtype=cp.int32), pot)
            (xg[gi], ug_[gi], gg_[gi]), _ag, _ug2 = nrl.jitter_accept(
                xg[gi], ug_[gi], gg_[gi], xfg, ufg, gfg, lag, accg,
                u_unif=uu[sl])
            assert np.array_equal(xm[sl].get(), xg[gi].get()), \
                f"S-9(iii-b) x mismatch step {n} group {gi}"
            assert np.array_equal(um[sl].get(), ug_[gi].get()), \
                f"S-9(iii-b) u mismatch step {n} group {gi}"
            assert np.array_equal(gm[sl].get(), gg_[gi].get()), \
                f"S-9(iii-b) g mismatch step {n} group {gi}"
    print("S-9(iii-b) mixed-(L, eps) full step == per-group scalar full step: "
          "50 steps bitwise OK")
    print("S-9 PASS")


def _sadv_build(tmp):
    """Minimal valid registered matrix + artifacts for the adversarial
    validator regressions (fix-verification defects 1-3)."""
    import copy
    gold, _sha = load_gold()
    prov = {**triple_hash(), "cupy_backend": PRODUCTION_BACKEND,
            "argv": "sadv-fixture"}
    manifest = {**triple_hash(), "created_utc": "t", "note": "sadv"}
    (tmp / "lock.json").write_text(json.dumps(manifest, sort_keys=True))
    side = tmp / "side"
    audit = tmp / "audit"
    side.mkdir()
    audit.mkdir()
    for name in S7_AUDIT_ARTIFACTS:
        np.savez_compressed(audit / name,
                            p0=np.zeros(S7_P0_SHAPES[name], dtype=np.float32),
                            provenance=json.dumps(prov, sort_keys=True))
    grid = TRAJ_STEPS_REGISTERED
    rows = []
    for ds in DATASETS:
        reps = CELL_REPS[ds]
        for prior in PRIORS:
            dim = len(gold[(ds, prior)]["gold_mean"])
            gid = gold[(ds, prior)]["gold_id"]
            for rep in range(3):
                base = seed_base(ds, prior, rep)
                tag = f"{ds}_p{prior}_rep{rep}"
                u_c = draw_rotations(base + STREAMS["rot_closed"], reps)
                probe = np.zeros(dim)
                probe[3] = 1.0
                np.savez_compressed(side / f"chees_design_{tag}.npz",
                                    provenance=json.dumps(prov, sort_keys=True),
                                    u_rot=u_c,
                                    u_replay=draw_rotations(
                                        base + STREAMS["rot_replay"], reps),
                                    u_ref=draw_rotations(
                                        base + STREAMS["rot_reference"], REF_REPS),
                                    probe=probe,
                                    x0=np.zeros((reps, K_CHAINS, dim), np.float32))
                df = design_fields(ds, prior, rep, dim, reps, 0.5, 0.75, False)
                common = {"windows": 150, "accept": {"warmup": 0.7, "equil": 0.7,
                                                     "scored": 0.7},
                          "ug": {"warmup": 1, "equil": 1, "scored": 1},
                          "bp": {"warmup": 1, "equil": 1, "scored": 1},
                          "wc": {"warmup": 1.0, "equil": 1.0, "scored": 1.0},
                          "mean_l": {"warmup": 8.0, "equil": 10.0, "scored": 10.0},
                          "elapsed_sec": 1.0, "provenance": prov,
                          "gold_id": gid, "gold_sha256": GOLD_PIN,
                          "trace_var": 1.0, "single_trace_var": 1.0,
                          "mse": 0.02, "single_mse": 0.1, "bias_sq": 0.001,
                          "gold_mse": 0.02}
                qcc = {ph: {"t_clamp": 0, "eps_clamp": 0, "l_max": 0,
                            "divergent": 0, "all_zero_weight": 0,
                            "proposals": 100, "ensemble_steps": 100}
                       for ph in ("warmup", "equil", "scored")}
                for arm, coup in ARMS:
                    t_x = np.full(reps, 3.0)
                    row = {**df, **common, "kind": "closed_loop", "arm": arm,
                           "coupling": coup, "method": coup, "variant": "primary",
                           "dyn_seed": base + STREAMS[f"arm_{arm}"],
                           "rot_seed": base + STREAMS["rot_closed"],
                           "steps": N_TOTAL, "n_warm": N_WARM,
                           "n_equil": N_EQUIL, "n_score": N_SCORE,
                           "group_size": 1, "T_star": t_x.tolist(),
                           "eps_star": np.full(reps, 0.5).tolist(),
                           "T_star_cf": (t_x * 1.02).tolist(),
                           "freeze_clamp": 0, "frozenness_ok": True,
                           "warmup_end_abar_mean": 0.65,
                           "warmup_end_abar_median": 0.65,
                           "scored_abar_mean": 0.65, "qc_counters": qcc,
                           "m_probe": np.zeros(reps).tolist(),
                           "m_probe_freeze": np.zeros(reps).tolist(),
                           "mean_realized_tlen": np.full(reps, 5.1).tolist(),
                           "tuner_traj": {"step": list(grid),
                                          **{k: [1.0] * len(grid)
                                             for k in TRAJ_KEYS[1:]}},
                           "halton_index_ranges": {"warmup": [1, N_WARM],
                                                   "equil": [N_WARM + 1,
                                                             N_WARM + N_EQUIL],
                                                   "scored": [N_WARM + N_EQUIL + 1,
                                                              N_TOTAL]}}
                    rows.append(row)
                    pay = {"provenance": json.dumps(prov, sort_keys=True),
                           "u_rot": u_c,
                           "T_star": np.asarray(row["T_star"]),
                           "eps_star": np.asarray(row["eps_star"])}
                    for k in ("traj_T", "traj_eps", "traj_ghat", "traj_ghat_cf",
                              "traj_abar"):
                        pay[k] = np.ones((N_WARM, reps))
                    np.savez_compressed(side / f"chees_tuner_{tag}_arm{arm}.npz",
                                        **pay)
                rows.append({**df, **common, "kind": "reference",
                             "arm": "reference", "coupling": "iid",
                             "method": "iid", "variant": "primary",
                             "dyn_seed": base + STREAMS["reference"],
                             "rot_seed": base + STREAMS["rot_reference"],
                             "windows": REF_STEPS - REF_BURN, "steps": REF_STEPS,
                             "burnin": REF_BURN, "reps_ref": REF_REPS,
                             "chains_ref": REF_CHAINS,
                             "accept": {"burnin": 0.7, "scored": 0.7},
                             "qc_counters": {ph: {"l_max": 0, "divergent": 0,
                                                  "proposals": 100}
                                             for ph in ("burnin", "scored")},
                             "halton_index_ranges": {"burnin": [1, REF_BURN],
                                                     "scored": [REF_BURN + 1,
                                                                REF_STEPS]}})
                for src, coup in REPLAY_RUNS:
                    nm = f"replay_{src}_{coup}"
                    rows.append({**df, **common, "kind": "replay", "arm": nm,
                                 "coupling": coup, "method": coup,
                                 "variant": "primary", "source_arm": src,
                                 "dyn_seed": base + STREAMS[nm],
                                 "rot_seed": base + STREAMS["rot_replay"],
                                 "x0_seed": base + STREAMS["replay_x0"],
                                 "steps": REPLAY_TOTAL, "burnin": REPLAY_BURN,
                                 "accept": {"burnin": 0.7, "scored": 0.7},
                                 "qc_counters": {ph: {"l_max": 0, "divergent": 0,
                                                      "proposals": 100}
                                                 for ph in ("burnin", "scored")},
                                 "halton_index_ranges": {"burnin": [1, REPLAY_BURN],
                                                         "scored": [REPLAY_BURN + 1,
                                                                    REPLAY_TOTAL]}})
    return rows, tmp / "lock.json", side, audit


def selftest_validator_adversarial():
    """Fix-verification regressions: (i) the valid canonical 144-row matrix
    passes clean; (ii) a 145th row with an unregistered variant ABORTS;
    (iii) an S-7 artifact renamed to _gpu but carrying stub backend
    provenance ABORTS on provenance; (iv) schema-vs-divergence: missing
    T_star_cf ABORTS, matched sampler NaNs validate as DIVERGED (not a
    mismatch abort); (v) 54 G4 rows without the K1 trigger ABORT."""
    import copy
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="chees_sadv_",
                                dir=os.environ.get("TMPDIR", "/tmp")))
    try:
        rows, lock, side, audit = _sadv_build(tmp)

        def expect_abort(tag, needle, rows_x):
            try:
                validate_campaign(rows_x, lock, side, audit, full=True)
            except (AssertionError, SystemExit) as exc:
                assert needle in str(exc), f"Sadv {tag}: wrong abort: {exc}"
                return
            raise AssertionError(f"Sadv {tag}: attack was ACCEPTED")

        _m, _g, _s, div = validate_campaign(rows, lock, side, audit, full=True)
        assert div == [], f"Sadv(i): clean matrix produced flags {div}"
        extra = copy.deepcopy(rows[7])
        extra["variant"] = "unregistered-extra"
        expect_abort("(ii) 145th-row", "unregistered variant", rows + [extra])
        gz = audit / S7_AUDIT_ARTIFACTS[0]
        keep = gz.read_bytes()
        prov_stub = {**triple_hash(), "cupy_backend": "numpy-stub"}
        np.savez_compressed(gz, p0=np.zeros(S7_P0_SHAPES[S7_AUDIT_ARTIFACTS[0]],
                                            dtype=np.float32),
                            provenance=json.dumps(prov_stub, sort_keys=True))
        expect_abort("(iii) stub-backend rename", "BACKEND ABORT", rows)
        gz.write_bytes(keep)
        r2 = copy.deepcopy(rows)
        victim = next(r for r in r2 if r["kind"] == "closed_loop")
        del victim["T_star_cf"]
        expect_abort("(iv-a) missing array", "mandatory field T_star_cf", r2)
        r3 = copy.deepcopy(rows)
        victim = next(r for r in r3 if r["kind"] == "closed_loop")
        victim["T_star"] = list(victim["T_star"])
        victim["T_star"][0] = float("nan")
        sc = side / (f"chees_tuner_{victim['dataset']}_p{victim['prior']}_rep"
                     f"{victim['replicate']}_arm{victim['arm']}.npz")
        with np.load(sc, allow_pickle=True) as d:
            pay = {k: d[k] for k in d.files}
        pay["T_star"] = np.asarray(victim["T_star"])
        np.savez_compressed(sc, **pay)
        _m, _g, _s, div = validate_campaign(r3, lock, side, audit, full=True)
        assert any("T_star" in f[4] and f[3] == "primary" for f in div), \
            "Sadv(iv-b): matched NaN did not flag DIVERGED (primary variant)"
        # restore the NaN sidecar from (iv-b) so later checks see clean primaries
        with np.load(sc, allow_pickle=True) as d:
            pay = {k: d[k] for k in d.files}
        pay["T_star"] = np.asarray(
            next(r for r in rows if r["kind"] == "closed_loop")["T_star"])
        np.savez_compressed(sc, **pay)

        def make_g4_rows(base_rows):
            g4_rows = []
            for r in base_rows:
                if r["kind"] != "closed_loop":
                    continue
                g = copy.deepcopy(r)
                g["variant"] = "g4"
                g["group_size"] = G4_GROUP
                g["T_star"] = np.full(r["reps"] // G4_GROUP, 3.0).tolist()
                g["eps_star"] = np.full(r["reps"] // G4_GROUP, 0.5).tolist()
                g["T_star_cf"] = None
                g["dyn_seed"] = r["seed_base"] + STREAMS[f"g4_{r['arm']}"]
                g["rot_seed"] = r["seed_base"] + STREAMS["rot_g4"]
                g4_rows.append(g)
            return g4_rows

        expect_abort("(v) untriggered G4", "trigger did NOT fire",
                     rows + make_g4_rows(rows))
        # (vi) closing-check residual 4: triggered 54-row matrix with a
        # matched G4-only eps_star NaN, run THROUGH make_table -- the primary
        # ladder must be unaffected and the G4 report must print exactly one
        # DIVERGED line with no partial statistics.
        import contextlib
        import io
        r6 = copy.deepcopy(rows)
        wide = None
        for r in r6:
            if r["kind"] == "closed_loop" and r["arm"] == "A":
                reps = r["reps"]
                wide = ([1.0] * (reps // 2) + [8.0] * (reps - reps // 2))
                r["T_star"] = list(wide)                 # IQR(log2) = 3 > 1: K1
                nm = (f"chees_tuner_{r['dataset']}_p{r['prior']}_rep"
                      f"{r['replicate']}_armA.npz")
                with np.load(side / nm, allow_pickle=True) as d:
                    pay = {k: d[k] for k in d.files}
                pay["T_star"] = np.asarray(wide)
                np.savez_compressed(side / nm, **pay)
        g4_rows = make_g4_rows(r6)
        poison = next(g for g in g4_rows
                      if (g["dataset"], g["prior"], g["replicate"], g["arm"])
                      == ("wdbc", 0.1, 1, "B"))
        poison["eps_star"] = list(poison["eps_star"])
        poison["eps_star"][0] = float("nan")             # G4-ONLY divergence
        prov6 = {**triple_hash(), "cupy_backend": PRODUCTION_BACKEND}
        for g in g4_rows:
            base = g["seed_base"]
            u_g, u_f = draw_rotations_g4(base + STREAMS["rot_g4"],
                                         g["reps"] // G4_GROUP, g["reps"])
            pay = {"provenance": json.dumps(prov6, sort_keys=True),
                   "u_rot": u_g, "u_fork": u_f,
                   "T_star": np.asarray(g["T_star"]),
                   "eps_star": np.asarray(g["eps_star"])}
            for k in ("traj_T", "traj_eps", "traj_ghat", "traj_ghat_cf",
                      "traj_abar"):
                pay[k] = np.ones((N_WARM, g["reps"] // G4_GROUP))
            np.savez_compressed(
                side / (f"chees_tuner_{g['dataset']}_p{g['prior']}_rep"
                        f"{g['replicate']}_arm{g['arm']}_g4.npz"), **pay)
        rows_path = tmp / "rows_vi.jsonl"
        with rows_path.open("w") as f:
            for r in r6 + g4_rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            make_table([rows_path], tmp / "t_vi.tex", qc_only=False,
                       lock_path=lock, sidecar_dir=side, audit_dir=audit)
        out6 = buf.getvalue()
        assert "UNRESOLVED (K1" in out6, "Sadv(vi): K1 reading missing"
        assert "QC fail=False" in out6, \
            "Sadv(vi): G4-only divergence contaminated the primary QC state"
        assert "DIVERGED (fails every gate" not in out6, \
            "Sadv(vi): a primary cell was marked DIVERGED by G4-only divergence"
        div_lines = [ln for ln in out6.splitlines()
                     if "[clustered" in ln and "DIVERGED" in ln]
        assert len(div_lines) == 1 and "wdbc/p=0.1" in div_lines[0], \
            f"Sadv(vi): expected exactly one G4 DIVERGED line, got {div_lines}"
        assert "B/A=" not in div_lines[0], \
            "Sadv(vi): partial statistics leaked into the G4 DIVERGED line"
        n_stats = [ln for ln in out6.splitlines()
                   if "[clustered" in ln and "B/A=" in ln]
        assert len(n_stats) == 5, \
            f"Sadv(vi): expected 5 finite G4 context cells, got {len(n_stats)}"
        print("Sadv PASS: canonical matrix clean; 145th-row attack aborts; "
              "stub-provenance _gpu rename aborts; missing mandatory array "
              "aborts while matched sampler NaNs -> DIVERGED; untriggered "
              "G4 rows abort; triggered matrix with G4-only eps_star NaN -> "
              "primary ladder isolated (UNRESOLVED K1, QC fail=False) + one "
              "G4 DIVERGED context line, no partial statistics")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TEST_ORDER = ["trio", "Sgrad", "S2", "S3a", "S3b", "S4", "S5", "S6", "S7",
              "S8", "S9", "S10", "Sadv"]


def run_selftests(args):
    th = triple_hash()
    print("== chees_real_logistic selftest log (V3-6 triple-hash header) ==")
    for key in ("chees_script_sha256", "nuts_substrate_sha256", "precommit_sha256"):
        print(f"{key} = {th[key]}")
    backend = ("NUMPY-STUB CPU SMOKE -- registered GPU selftests DEFERRED"
               if IS_STUB else "cupy (GPU)")
    print(f"backend: {backend}")
    if args.tests:
        sel = set(args.tests.split(","))
        unknown = sel - set(TEST_ORDER)
        if unknown:
            raise SystemExit(f"unknown --tests selector(s) {sorted(unknown)}; "
                             f"valid: {','.join(TEST_ORDER)} (audit ruling 9)")
    else:
        sel = set(TEST_ORDER)
    fns = {"trio": selftest_trio, "Sgrad": selftest_gradient_totality,
           "S2": selftest_s2, "S3a": selftest_s3a,
           "S3b": selftest_s3b, "S4": selftest_s4, "S5": selftest_s5,
           "S6": selftest_s6,
           "S7": lambda: selftest_s7(args.data_dir, args.selftest_audit_dir),
           "S8": selftest_s8, "S9": selftest_s9,
           "S10": s10_seed_uniqueness,
           "Sadv": selftest_validator_adversarial}
    for name in TEST_ORDER:
        if name not in sel:
            continue
        t0 = time.perf_counter()
        fns[name]()
        print(f"[{name}] completed in {time.perf_counter() - t0:.1f}s", flush=True)
    print("selftest suite COMPLETE" + (" (CPU-STUB SMOKE; GPU RERUN REQUIRED "
          "BEFORE THE EXECUTABLE HASH LOCK)" if IS_STUB else ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--tests", default=None,
                        help="comma subset of " + ",".join(TEST_ORDER))
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--make-table", action="store_true")
    parser.add_argument("--qc-only", action="store_true",
                        help="V2-10 blinded replicate-0 mode (diagnostics only)")
    parser.add_argument("--table-inputs", nargs="+", type=Path,
                        default=[Path(f"outputs/sweeps/chees_real_rep{r}.jsonl")
                                 for r in range(3)])
    parser.add_argument("--table-out", type=Path,
                        default=Path("paper/tables/chees_closedloop.tex"))
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--priors", nargs="+", type=float, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--log-full-tuner", action="store_true")
    parser.add_argument("--sidecar-dir", type=Path,
                        default=Path("outputs/sweeps/chees_sidecars"))
    parser.add_argument("--g4-stage", action="store_true",
                        help="TRIGGER-ONLY G=4 second stage (V2-2/audit 5): "
                             "validates primary rows, confirms the arm-A "
                             "trigger, then runs ONLY G4 arms A/B/C")
    parser.add_argument("--selftest-audit-dir", type=Path,
                        default=Path("outputs/selftests"))
    parser.add_argument("--lock-manifest", type=Path,
                        default=LOCK_MANIFEST_DEFAULT,
                        help="immutable executable-hash-lock manifest "
                             "(audit finding 4)")
    parser.add_argument("--write-lock-manifest", action="store_true",
                        help="create the lock manifest (executable-hash-lock "
                             "step ONLY; refuses overwrite)")
    args = parser.parse_args()
    if args.write_lock_manifest:
        write_lock_manifest(args.lock_manifest)
        return
    if args.selftest:
        run_selftests(args)
        return
    if args.print_plan:
        print_plan()
        return
    if args.make_table:
        make_table(args.table_inputs, args.table_out, qc_only=args.qc_only,
                   lock_path=args.lock_manifest, sidecar_dir=args.sidecar_dir,
                   audit_dir=args.selftest_audit_dir)
        return
    if args.g4_stage:
        g4_stage(args)
        return
    if args.out is None:
        parser.error("--out is required unless --selftest/--print-plan/"
                     "--make-table/--g4-stage/--write-lock-manifest")
    campaign(args)


if __name__ == "__main__":
    main()



