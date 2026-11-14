"""P4 gate analysis: registered V3-4/V4-1/V4-2 gates + V3-1 hardened gold
+ amendment-v2E shift/spread diagnostics + S2b b_c/R_c.

Implements EXACTLY the analysis registered in
campaigns/screens_precommit.md (sections V3-1, V3-3, V3-4,
V4-1, V4-2, amendment v2 sections C and E, and the 2026-07-18 remediation
ruling), on the P4 gold artifact
outputs/sweeps/rr_extrapolation_gold_p4_fd354ef.jsonl and its registered
series sidecars.  CPU-only; read-only on all accepted artifacts; writes
only the machine-readable gate report (default
outputs/screens/p4_gate_report.json).

Per cell (wdbc/spambase/adult x lambda in {0.1, 0.3}; passing attempt
selected per the remediation ruling -- auto prefers attempt-1 rows when a
complete, integrity-checked attempt-1 row set exists for the cell):
  (1) V4-2 Rhat/ESS gates per arm, RECOMPUTED from the sidecar per-step
      ensemble-mean series with the committed rank-normalized estimators
      (ported byte-verbatim from scripts/rr_extrapolation.py; anti-drift
      checked in --selftest): Rhat = max(rank-split, folded-rank-split)
      over the 3 seed-series split in half (6 half-chains) <= 1.01; rank-
      normalized bulk ESS >= 400; gates bind on the control (zero) and
      primary dispersed (chol) arms; the gauss comparability arm is
      reported descriptively.
  (2) Acceptance gate: post-burn MALA acceptance in [0.45, 0.80] per
      seed/arm (zero + chol binding; gauss descriptive).
  (3) Two Bonferroni families (V3-4 + V4-1), q_P4 = Phi^{-1}(1 -
      0.05/(2*388)): (i) split-half within the dispersed arm (halves
      pooled across seeds; per-half MCSE from the recorded per-half batch
      means); (ii) control-vs-dispersed cross-arm (per-arm MCSE from the
      recorded pooled batch means).  MCSE_combined = sqrt(MCSE_a^2 +
      MCSE_b^2).  Gates per family, per coordinate: precision
      MCSE_combined/SD <= 0.02 and the SOLE agreement gate |Delta| +
      q_P4*MCSE_combined <= 0.10*SD, with SD_cj the Bessel-corrected
      marginal SD of the INDIVIDUAL retained draws (the K*kept individual
      chain draws) of the successful primary dispersed attempt.  Those
      draws were never archived (only the per-step ensemble-mean series
      was); SD_cj is therefore ESTIMATED as sqrt(K) * BesselSD(series
      draws, pooled over seeds), exact at stationarity because the K gold
      chains are mutually independent (Var(ensemble mean) = Var(draw)/K).
      The stricter literal series-SD reading is reported per cell as a
      sensitivity block (it is not gate-bearing; under it the precision
      gate would require series-ESS ~2500 and be unsatisfiable by design
      at the registered V4-2 floor of 400).  |Delta| <= q*MCSE
      nonrejection counts are DESCRIPTIVE ONLY.
  (4) mu_gold_hard (pooled chol mean over 3 seeds) and r95_c =
      sqrt(sum_j (q_P4 * MCSE_cj)^2)  (V3-1 diagonal/Bonferroni radius).
  (5) shift_sq = ||gold_new - gold_old||^2 / old_gold_spread and the
      DESCRIPTIVE 3x spread ratio vs the archived common-start gold.
  (6) S2b: b_c = min over q in {VI, converged EP} of ||mu_q -
      mu_gold_hard||; R_c = max(0, b_c - r95_c)^2 / r95_c^2; G_c PENDING
      until the V3-5 rescore (gold_rescore_p4.jsonl) lands.
  (7) P4_ALL6_STABLE and the pending-G_c ALT1_GO skeleton.

--selftest validates every formula on synthetic fixtures (no accepted
artifact is read) and byte-compares the ported estimator functions
against scripts/rr_extrapolation.py.
"""

import argparse
import ast
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RR_SCRIPT = REPO_ROOT / "scripts" / "rr_extrapolation.py"
RR_SCRIPT_LOCKED_SHA = "26030537ad5254e46790150cd28d19c628f9fd4281526fa1a6a9caac9c23dbd3"

GOLD_DEFAULT = REPO_ROOT / "outputs/sweeps/rr_extrapolation_gold_p4_fd354ef.jsonl"
ARCHIVED_GOLD_DEFAULT = REPO_ROOT / "outputs/sweeps/rr_extrapolation_gold.jsonl"
VI_DEFAULT = REPO_ROOT / "outputs/sweeps/vi_baseline.jsonl"
RESCORE_PATH = REPO_ROOT / "outputs/sweeps/gold_rescore_p4.jsonl"
REPORT_DEFAULT = REPO_ROOT / "outputs/screens/p4_gate_report.json"

P4_SCHEMA_VERSION = "p4-gold-1"
DATASETS = ["wdbc", "spambase", "adult"]
PRIORS = [0.1, 0.3]
CELLS = [(ds, pr) for ds in DATASETS for pr in PRIORS]  # cell_idx order (V3-3)
D_P4 = 388
Q_P4_REGISTERED = 3.8286158286  # Phi^{-1}(1 - 0.05/(2*388)), registered decimal
RHAT_MAX = 1.01
ESS_MIN = 400.0
ACC_LO, ACC_HI = 0.45, 0.80
PREC_CEIL = 0.02   # V4-1 (a): MCSE_combined / SD <= 0.02
EQUIV_CEIL = 0.10  # V4-1 (b): |Delta| + q*MCSE_combined <= 0.10*SD (SOLE agreement gate)
N_SEEDS = 3
BATCH_LEN = 1000
ARMS = {0: "zero", 1: "chol", 2: "gauss"}
GATED_ARMS = (0, 1)  # control + primary dispersed; gauss (2) descriptive only

# --- V4-2 diagnostics + V3-1 MCSE, ported BYTE-VERBATIM from the locked
# scripts/rr_extrapolation.py (sha256 26030537ad5254e4..., Amendment v7),
# which itself carries the Amendment-v6-pinned copy from
# scripts/s3_poisson_skew_screen.py (sha256 dcee5eb611b7...).  Do not edit;
# --selftest byte-compares each function's source against the committed
# rr_extrapolation.py copy (anti-drift check, v7 pattern).

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


def p4_mcse(pooled_batch_means):
    """V3-1: MCSE_j = sd(all pooled batch means_j) / sqrt(n_batches_total)."""
    bm = np.asarray(pooled_batch_means, dtype=np.float64)
    return bm.std(axis=0, ddof=1) / math.sqrt(bm.shape[0])


PORTED_FUNCS = ["_import_diag_deps", "_split", "_rank_normal", "_rhat",
                "_bulk_ess", "diagnostics", "p4_mcse"]


# --- registered formula helpers (unit-tested in --selftest) -----------------

def q_p4():
    _import_diag_deps()
    q = float(ndtri(1.0 - 0.05 / (2.0 * D_P4)))
    if abs(round(q, 10) - Q_P4_REGISTERED) > 1e-12:
        raise SystemExit(f"computed q_P4 {q!r} does not round to the registered "
                         f"{Q_P4_REGISTERED}")
    return q


def combined_mcse(mcse_a, mcse_b):
    """V3-4: MCSE_combined = sqrt(MCSE_a^2 + MCSE_b^2), per coordinate."""
    return np.sqrt(np.asarray(mcse_a) ** 2 + np.asarray(mcse_b) ** 2)


def precision_gate(mcse_combined, sd):
    """V4-1 (a): MCSE_combined,cj / SD_cj <= 0.02, all coordinates."""
    ratio = np.asarray(mcse_combined) / np.asarray(sd)
    ok = bool(np.all(np.isfinite(ratio)) and np.all(ratio <= PREC_CEIL))
    return ok, ratio


def equivalence_gate(delta, mcse_combined, sd, q):
    """V4-1 (b), SOLE agreement gate:
    |Delta_cj| + q_P4 * MCSE_combined,cj <= 0.10 * SD_cj, all coordinates."""
    lhs = np.abs(np.asarray(delta)) + q * np.asarray(mcse_combined)
    rhs = EQUIV_CEIL * np.asarray(sd)
    margin = lhs / rhs  # <= 1 passes
    ok = bool(np.all(np.isfinite(margin)) and np.all(lhs <= rhs))
    return ok, margin


def z_nonreject_descriptive(delta, mcse_combined, q):
    """DESCRIPTIVE ONLY (V4-1): count of coordinates with |Delta| > q*MCSE_c."""
    return int(np.sum(np.abs(np.asarray(delta)) > q * np.asarray(mcse_combined)))


def acceptance_ok(accepts_postburn):
    """V3-4: post-burn acceptance in [0.45, 0.80] per seed (inclusive)."""
    a = np.asarray(accepts_postburn, dtype=np.float64)
    return bool(a.size == N_SEEDS and np.all(np.isfinite(a))
                and np.all(a >= ACC_LO) and np.all(a <= ACC_HI))


def r95_from_mcse(mcse, q):
    """V3-1: r95_c = sqrt( sum_j (q_P4 * MCSE_cj)^2 )."""
    m = np.asarray(mcse, dtype=np.float64)
    if not np.all(np.isfinite(m)):
        return float("nan")
    return float(np.sqrt(np.sum((q * m) ** 2)))


def s2b_r(b, r95):
    """S2b: R_c = max(0, b_c - r95_c)^2 / r95_c^2."""
    return float(max(0.0, b - r95) ** 2 / r95 ** 2)


def shift_sq_stat(gold_new, gold_old, old_spread):
    """Amendment v2 E: shift_sq = ||gold_new - gold_old||^2 / old_gold_spread."""
    d = np.asarray(gold_new, dtype=np.float64) - np.asarray(gold_old, dtype=np.float64)
    return float(np.sum(d ** 2) / old_spread)


def split_half_family(seed_batch_means):
    """V3-4 split-half construction within one arm: per seed, first half vs
    second half of kept draws (batch boundaries align exactly); halves pooled
    across seeds; per-half MCSE from that half's pooled batch means (V3-1
    estimator).  Returns (delta, mcse_a, mcse_b, mu_a, mu_b)."""
    bm = np.asarray(seed_batch_means, dtype=np.float64)  # (seeds, nb, d)
    nb = bm.shape[1]
    if nb % 2 != 0:
        raise SystemExit(f"odd batch count {nb}: halves would not align")
    half = nb // 2
    first = bm[:, :half, :].reshape(-1, bm.shape[2])
    second = bm[:, half:, :].reshape(-1, bm.shape[2])
    mu_a, mu_b = first.mean(axis=0), second.mean(axis=0)
    return mu_a - mu_b, p4_mcse(first), p4_mcse(second), mu_a, mu_b


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- loaders (fail-closed; duplicate keys ABORT, never last-win) ------------

def load_jsonl_tolerant(path, warnings):
    """Parse a JSONL file.  A malformed FINAL nonblank line is skipped with a
    mid-write warning (registered mid-write tolerance); any malformed earlier
    line aborts."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    nonblank = [(i, ln) for i, ln in enumerate(lines) if ln.strip()]
    rows = []
    for pos, (i, ln) in enumerate(nonblank):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if pos == len(nonblank) - 1:
                warnings.append(f"{path}: final line {i + 1} is not valid JSON -- "
                                "treated as a mid-write partial row and skipped")
            else:
                raise SystemExit(f"{path}: malformed interior line {i + 1}")
    return rows


def index_p4_rows(rows, path="<rows>"):
    """Key rows by (cell_idx, arm_idx, attempt); schema check; dup -> ABORT."""
    idx = {}
    for r in rows:
        if r.get("schema_version") != P4_SCHEMA_VERSION:
            raise SystemExit(f"{path}: row gold_id={r.get('gold_id')} has "
                             f"schema_version={r.get('schema_version')!r}, "
                             f"expected {P4_SCHEMA_VERSION!r}")
        key = (int(r["cell_idx"]), int(r["arm_idx"]), int(r["attempt"]))
        if key in idx:
            raise SystemExit(f"{path}: duplicate row for (cell, arm, attempt)="
                             f"{key} -- loader aborts (never last-win)")
        expect_ds, expect_pr = CELLS[key[0]]
        if r["dataset"] != expect_ds or float(r["prior"]) != expect_pr:
            raise SystemExit(f"{path}: cell_idx {key[0]} carries "
                             f"{r['dataset']}/{r['prior']}, expected "
                             f"{expect_ds}/{expect_pr}")
        idx[key] = r
    return idx


def parse_attempt_map(text):
    """--attempt-map 'auto' or 'cell:att[,cell:att...]' (e.g. '5:1')."""
    if text.strip().lower() == "auto":
        return None
    out = {}
    for part in text.split(","):
        c, a = part.strip().split(":")
        c, a = int(c), int(a)
        if c not in range(6) or a not in (0, 1):
            raise SystemExit(f"--attempt-map entry {part!r} out of range")
        if c in out:
            raise SystemExit(f"--attempt-map repeats cell {c}")
        out[c] = a
    return out


def select_attempts(idx, attempt_map, integrity_fn, notes, cells=range(6)):
    """Per cell: the attempt whose rows the analysis evaluates.
    Explicit --attempt-map wins (missing rows abort).  Auto: prefer attempt 1
    when BOTH gated arms (zero, chol) have attempt-1 rows passing sidecar
    integrity; a partial/incomplete attempt-1 falls back to attempt 0 with a
    provisional note (mid-write tolerance)."""
    chosen = {}
    for cell in cells:
        if attempt_map is not None and cell in attempt_map:
            att = attempt_map[cell]
            for arm in GATED_ARMS:
                if (cell, arm, att) not in idx:
                    raise SystemExit(f"--attempt-map forces cell {cell} attempt "
                                     f"{att} but arm {arm} row is absent")
            chosen[cell] = att
            continue
        att1_arms = [arm for arm in GATED_ARMS if (cell, arm, 1) in idx]
        if len(att1_arms) == len(GATED_ARMS):
            bad = [arm for arm in GATED_ARMS if not integrity_fn(idx[(cell, arm, 1)])]
            if not bad:
                chosen[cell] = 1
                continue
            notes.append(f"cell {cell}: attempt-1 rows present but sidecar "
                         f"integrity failed on arm(s) {bad} (mid-write?) -- "
                         "falling back to attempt 0 (provisional)")
        elif att1_arms:
            notes.append(f"cell {cell}: attempt-1 rows present only for arm(s) "
                         f"{att1_arms} (mid-write?) -- falling back to attempt 0 "
                         "(provisional)")
        for arm in GATED_ARMS:
            if (cell, arm, 0) not in idx:
                raise SystemExit(f"cell {cell}: no attempt-0 row for gated arm {arm}")
        chosen[cell] = 0
    return chosen


def sidecar_path(row):
    p = Path(row["series_sidecar"])
    return p if p.is_absolute() else REPO_ROOT / p


def sidecar_integrity(row):
    p = sidecar_path(row)
    return p.exists() and sha256_file(p) == row["series_sidecar_sha256"]


def load_series(row, warnings):
    """Load + validate one sidecar: sha256 vs the row, metadata vs the row,
    shape (N_SEEDS, kept, dim).  Also cross-checks that the recorded batch
    means / seed means are the batch/overall means of this series."""
    p = sidecar_path(row)
    sha = sha256_file(p)
    if sha != row["series_sidecar_sha256"]:
        raise SystemExit(f"sidecar sha mismatch for {p.name}: {sha[:12]} vs "
                         f"recorded {row['series_sidecar_sha256'][:12]}")
    z = np.load(p)
    for field, want in [("gold_id", row["gold_id"]), ("dataset", row["dataset"]),
                        ("cell_idx", row["cell_idx"]), ("arm_idx", row["arm_idx"]),
                        ("attempt", row["attempt"]), ("kept", row["kept"]),
                        ("batch_len", row["batch_len"])]:
        got = z[field].item() if z[field].shape == () else z[field]
        if str(got) != str(want):
            raise SystemExit(f"sidecar {p.name}: field {field}={got!r} vs row {want!r}")
    series = np.asarray(z["series"], dtype=np.float64)
    if series.shape != (N_SEEDS, int(row["kept"]), int(row["dim"])):
        raise SystemExit(f"sidecar {p.name}: series shape {series.shape} vs "
                         f"expected {(N_SEEDS, row['kept'], row['dim'])}")
    if not np.all(np.isfinite(series)):
        raise SystemExit(f"sidecar {p.name}: nonfinite series values")
    # cross-checks (recorded row bookkeeping vs the series)
    bm_rec = np.asarray(row["seed_batch_means"], dtype=np.float64)
    nb = int(row["n_batches_per_seed"])
    bm_re = series.reshape(N_SEEDS, nb, BATCH_LEN, -1).mean(axis=2)
    if not np.allclose(bm_rec, bm_re, rtol=1e-9, atol=1e-12):
        warnings.append(f"{p.name}: recorded batch means differ from series-"
                        f"recomputed by up to {np.max(np.abs(bm_rec - bm_re)):.3e}")
    sm_rec = np.asarray(row["seed_means"], dtype=np.float64)
    if not np.allclose(sm_rec, series.mean(axis=1), rtol=1e-9, atol=1e-12):
        warnings.append(f"{p.name}: recorded seed means differ from series means")
    return series


# --- per-cell analysis -------------------------------------------------------

def arm_block(row, series, q, gated, warnings):
    """Recompute V4-2 diagnostics from the series; cross-check vs recorded."""
    diag = diagnostics(series)
    rhat = np.asarray(diag["split_rhat"], dtype=np.float64)
    ess = np.asarray(diag["bulk_ess"], dtype=np.float64)
    d_rhat = float(np.max(np.abs(rhat - np.asarray(row["split_rhat"]))))
    d_ess = float(np.max(np.abs(ess - np.asarray(row["bulk_ess"]))))
    if d_rhat > 1e-9 or d_ess > 1e-6:
        warnings.append(f"{row['gold_id']}: recomputed diagnostics differ from "
                        f"recorded (max |dRhat|={d_rhat:.3e}, |dESS|={d_ess:.3e}); "
                        "gates use the recomputed values")
    finite = bool(np.all(np.isfinite(rhat)) and np.all(np.isfinite(ess)))
    diag_ok = bool(finite and np.max(rhat) <= RHAT_MAX and np.min(ess) >= ESS_MIN)
    acc = [float(a) for a in row["seed_accepts_postburn"]]
    acc_ok = acceptance_ok(acc)
    return {
        "arm": row["start_mode"], "attempt": int(row["attempt"]),
        "gold_id": row["gold_id"], "gated": gated,
        "rhat_max": float(np.max(rhat)), "ess_min": float(np.min(ess)),
        "recomputed_vs_recorded_max_abs_diff": {"rhat": d_rhat, "ess": d_ess},
        "accept_postburn": acc, "diag_ok": diag_ok, "acc_ok": acc_ok,
        "finite": finite,
    }, rhat, ess


def analyze_cell(cell, att, idx, archived, vi_rows, ep_rows, q, warnings):
    ds, pr = CELLS[cell]
    rz = idx[(cell, 0, att)]
    rc = idx[(cell, 1, att)]
    rg = idx.get((cell, 2, att))
    dim = int(rc["dim"])
    sz = load_series(rz, warnings)
    sc = load_series(rc, warnings)
    zb, _, _ = arm_block(rz, sz, q, True, warnings)
    cb, _, _ = arm_block(rc, sc, q, True, warnings)
    arms = {"zero": zb, "chol": cb}
    if rg is not None and sidecar_integrity(rg):
        sg = load_series(rg, warnings)
        gb, _, _ = arm_block(rg, sg, q, False, warnings)
        gb["descriptive_only"] = True
        arms["gauss"] = gb
        del sg
    else:
        arms["gauss"] = None
        warnings.append(f"cell {cell}: gauss comparability arm missing or failed "
                        f"integrity at attempt {att} (descriptive only; no gate impact)")

    # V4-1 SD_cj: Bessel-corrected marginal SD of the INDIVIDUAL retained
    # draws (K*kept chain draws) of the successful primary dispersed attempt.
    # The chain draws were never archived; the exact-at-stationarity estimator
    # is sqrt(K) * BesselSD(recorded ensemble-mean series, pooled over seeds)
    # since the K gold chains are mutually independent.  The literal series-SD
    # reading is kept as a non-gate-bearing sensitivity value.
    k_chains = int(rc["chains"])
    sd_series = sc.reshape(-1, dim).std(axis=0, ddof=1)
    sd = math.sqrt(k_chains) * sd_series

    # family (i): split-half within the dispersed arm
    delta_s, mcse_a, mcse_b, mu_a, mu_b = split_half_family(rc["seed_batch_means"])
    half_rec_a = np.asarray(rc["seed_split_half_first"], dtype=np.float64).mean(axis=0)
    half_rec_b = np.asarray(rc["seed_split_half_second"], dtype=np.float64).mean(axis=0)
    if not (np.allclose(mu_a, half_rec_a, rtol=1e-9, atol=1e-12)
            and np.allclose(mu_b, half_rec_b, rtol=1e-9, atol=1e-12)):
        warnings.append(f"cell {cell}: pooled split-half means differ from the "
                        "recorded seed_split_half fields")
    comb_s = combined_mcse(mcse_a, mcse_b)
    prec_s_ok, prec_s = precision_gate(comb_s, sd)
    equiv_s_ok, marg_s = equivalence_gate(delta_s, comb_s, sd, q)
    _, prec_s_ser = precision_gate(comb_s, sd_series)
    _, marg_s_ser = equivalence_gate(delta_s, comb_s, sd_series, q)

    # family (ii): control vs dispersed cross-arm
    mu_z = np.asarray(rz["gold_mean"], dtype=np.float64)
    mu_c = np.asarray(rc["gold_mean"], dtype=np.float64)
    for row, mu in ((rz, mu_z), (rc, mu_c)):
        pooled = np.asarray(row["seed_means"], dtype=np.float64).mean(axis=0)
        if not np.allclose(mu, pooled, rtol=1e-9, atol=1e-12):
            warnings.append(f"{row['gold_id']}: gold_mean differs from the "
                            "pooled seed means")
    mcse_z = p4_mcse(np.asarray(rz["seed_batch_means"], dtype=np.float64)
                     .reshape(-1, dim))
    mcse_c = p4_mcse(np.asarray(rc["seed_batch_means"], dtype=np.float64)
                     .reshape(-1, dim))
    for row, m in ((rz, mcse_z), (rc, mcse_c)):
        if not np.allclose(m, np.asarray(row["mcse"]), rtol=1e-9, atol=1e-15):
            warnings.append(f"{row['gold_id']}: recomputed MCSE differs from recorded")
    delta_x = mu_z - mu_c
    comb_x = combined_mcse(mcse_z, mcse_c)
    prec_x_ok, prec_x = precision_gate(comb_x, sd)
    equiv_x_ok, marg_x = equivalence_gate(delta_x, comb_x, sd, q)
    _, prec_x_ser = precision_gate(comb_x, sd_series)
    _, marg_x_ser = equivalence_gate(delta_x, comb_x, sd_series, q)

    # V3-1 hardened gold + radius
    r95 = r95_from_mcse(mcse_c, q)

    # amendment v2 E: shift + descriptive spread ratio vs archived gold
    old = archived[(ds, pr)]
    shift_sq = shift_sq_stat(mu_c, old["gold_mean"], old["gold_spread"])
    spread_ratio = float(rc["gold_spread"] / old["gold_spread"])

    # S2b (b_c, R_c now; G_c pending the V3-5 rescore)
    vi = vi_rows[(ds, pr)]
    ep = ep_rows[(ds, pr)]
    b_vi = float(np.linalg.norm(np.asarray(vi["mu_vi"], dtype=np.float64) - mu_c))
    b_ep = None
    if bool(ep.get("converged")):
        b_ep = float(np.linalg.norm(np.asarray(ep["mu_ep"], dtype=np.float64) - mu_c))
    else:
        warnings.append(f"cell {cell}: EP row not converged -- b_c is VI-only "
                        "(registered narrowing)")
    b_c = b_vi if b_ep is None else min(b_vi, b_ep)
    r_c = s2b_r(b_c, r95) if np.isfinite(r95) and r95 > 0 else float("nan")

    finite_ok = bool(
        arms["zero"]["finite"] and arms["chol"]["finite"]
        and np.all(np.isfinite(sd)) and np.all(sd > 0)
        and np.all(np.isfinite(delta_s)) and np.all(np.isfinite(comb_s))
        and np.all(np.isfinite(delta_x)) and np.all(np.isfinite(comb_x))
        and np.isfinite(r95) and np.isfinite(shift_sq) and np.isfinite(b_c)
        and all(np.isfinite(a) for a in zb["accept_postburn"] + cb["accept_postburn"]))

    gates = {
        "diag_zero_ok": zb["diag_ok"], "diag_chol_ok": cb["diag_ok"],
        "acc_zero_ok": zb["acc_ok"], "acc_chol_ok": cb["acc_ok"],
        "split_precision_ok": prec_s_ok, "split_equivalence_ok": equiv_s_ok,
        "cross_precision_ok": prec_x_ok, "cross_equivalence_ok": equiv_x_ok,
        "finite_ok": finite_ok,
    }
    cell_pass = all(gates.values())
    diagnostics_pass = (zb["diag_ok"] and cb["diag_ok"]
                        and zb["acc_ok"] and cb["acc_ok"])
    return {
        "cell_idx": cell, "dataset": ds, "prior": pr, "dim": dim,
        "attempt_used": att, "arms": arms, "gates": gates, "cell_pass": cell_pass,
        "diagnostics_pass": diagnostics_pass,
        "mu_gold_hard": [float(v) for v in mu_c],
        "r95": r95,
        "mcse_chol_max": float(np.max(mcse_c)),
        "sd_convention": {
            "gate_bearing": f"sd_draws_est = sqrt(K={k_chains}) * BesselSD(series)",
            "sd_draws_est_min": float(np.min(sd)), "sd_draws_est_max": float(np.max(sd)),
            "sd_series_min": float(np.min(sd_series)),
            "sd_series_max": float(np.max(sd_series)),
        },
        "families": {
            "split_half": {
                "max_abs_delta": float(np.max(np.abs(delta_s))),
                "max_mcse_combined_over_sd": float(np.max(prec_s)),
                "max_equiv_margin": float(np.max(marg_s)),
                "z_reject_count_descriptive": z_nonreject_descriptive(delta_s, comb_s, q),
                "sensitivity_series_sd": {
                    "max_mcse_combined_over_sd": float(np.max(prec_s_ser)),
                    "max_equiv_margin": float(np.max(marg_s_ser)),
                    "note": "literal series-draws SD reading; NOT gate-bearing",
                },
            },
            "cross_arm": {
                "max_abs_delta": float(np.max(np.abs(delta_x))),
                "max_mcse_combined_over_sd": float(np.max(prec_x)),
                "max_equiv_margin": float(np.max(marg_x)),
                "z_reject_count_descriptive": z_nonreject_descriptive(delta_x, comb_x, q),
                "sensitivity_series_sd": {
                    "max_mcse_combined_over_sd": float(np.max(prec_x_ser)),
                    "max_equiv_margin": float(np.max(marg_x_ser)),
                    "note": "literal series-draws SD reading; NOT gate-bearing",
                },
            },
        },
        "shift_sq": shift_sq,
        "spread_ratio_chol_over_archived_descriptive": spread_ratio,
        "chol_spread": float(rc["gold_spread"]),
        "archived_spread": float(old["gold_spread"]),
        "s2b": {"b_vi": b_vi, "b_ep": b_ep, "b_c": b_c,
                "b_source": "vi" if b_c == b_vi and (b_ep is None or b_vi <= b_ep)
                            else "ep",
                "r95": r95, "R_c": r_c, "G_c": None,
                "G_c_status": "PENDING: outputs/sweeps/gold_rescore_p4.jsonl "
                              "absent (V3-5 rescore not yet run)"},
    }


# --- report + console --------------------------------------------------------

def run_analysis(args):
    warnings, notes, judgment_calls = [], [], []
    q = q_p4()
    rr_sha = sha256_file(RR_SCRIPT)
    if rr_sha != RR_SCRIPT_LOCKED_SHA:
        warnings.append(f"scripts/rr_extrapolation.py sha {rr_sha[:16]} != locked "
                        f"{RR_SCRIPT_LOCKED_SHA[:16]} (working-tree drift?)")

    rows = load_jsonl_tolerant(args.gold, warnings)
    idx = index_p4_rows(rows, str(args.gold))
    for r in idx.values():
        att, kept, burn = int(r["attempt"]), int(r["kept"]), int(r["burnin"])
        if kept != 30000 * (att + 1) or burn != 5000 * (att + 1) or \
                int(r["batch_len"]) != BATCH_LEN or \
                int(r["n_batches_total"]) != N_SEEDS * kept // BATCH_LEN:
            raise SystemExit(f"{r['gold_id']}: kept/burnin/batch bookkeeping "
                             "violates the locked attempt arithmetic")

    archived = {}
    for r in load_jsonl_tolerant(args.archived_gold, warnings):
        if r.get("kind") != "gold":
            continue
        key = (r["dataset"], float(r["prior"]))
        if key in archived:
            raise SystemExit(f"duplicate archived gold row for {key}")
        archived[key] = r
    for (c, a, t), r in idx.items():
        if r["frozen_gold_id"] != archived[(r["dataset"], float(r["prior"]))]["gold_id"]:
            raise SystemExit(f"{r['gold_id']}: frozen_gold_id does not match the "
                             "archived gold row")

    vi_rows, ep_rows = {}, {}
    for r in load_jsonl_tolerant(args.vi, warnings):
        key = (r["dataset"], float(r["prior"]))
        book = vi_rows if r.get("kind") == "vi_baseline" else \
            ep_rows if r.get("kind") == "ep_baseline" else None
        if book is None:
            continue
        if key in book:
            raise SystemExit(f"duplicate {r['kind']} row for {key}")
        book[key] = r
    missing = [k for k in CELLS if k not in vi_rows or k not in ep_rows]
    if missing:
        raise SystemExit(f"vi_baseline.jsonl missing VI/EP rows for {missing}")

    attempt_map = parse_attempt_map(args.attempt_map)
    chosen = select_attempts(idx, attempt_map, sidecar_integrity, notes)

    dims_check = sum(int(idx[(c, 1, chosen[c])]["dim"]) for c in range(6))
    if dims_check != D_P4:
        raise SystemExit(f"family size check failed: sum of dims {dims_check} != {D_P4}")

    judgment_calls += [
        "SD_cj (V4-1, 'Bessel-corrected marginal SD of the INDIVIDUAL retained "
        "draws of the successful primary dispersed attempt') is read as the "
        "marginal SD of the K*kept individual CHAIN draws (K=64), i.e. the "
        "posterior marginal SD -- 'INDIVIDUAL' contrasts with the ensemble-mean "
        "series, v2E states posterior-SD units ('fitted posterior RMS SDs are "
        "~1.9-3.25'), and V4-1's own headroom arithmetic presumes a passable "
        "precision gate (under the literal series-SD reading the 0.02 ceiling "
        "would require series-ESS ~2500 while V4-2 registers ESS >= 400, i.e. "
        "the gate system would be unsatisfiable by design).  The chain draws "
        "were never archived, so SD_cj is ESTIMATED as sqrt(K) * BesselSD("
        "recorded ensemble-mean series, pooled over the 3 seeds) -- exact at "
        "stationarity because the K gold chains are mutually independent "
        "(Var(ensemble mean) = Var(draw)/K).  The literal series-SD reading is "
        "reported per family as a non-gate-bearing sensitivity block.  THIS IS "
        "THE KEY INTERPRETIVE CALL OF THIS ANALYSIS and is flagged for the "
        "result review that gates any S2b/stretch-matrix consequence.",
        "V4-2 Rhat/ESS and V3-4 acceptance gates bind on the control (zero) and "
        "primary dispersed (chol) arms per V4-2's 'control and dispersed "
        "separately'; the gauss comparability arm is computed and reported "
        "descriptively only.",
        "Split-half per-half MCSEs use the V3-1 batch-means estimator on each "
        "half's recorded batch means (batch boundaries align exactly with the "
        "half boundary); halves pooled across seeds before testing per V3-4.",
        "All gate comparisons are inclusive at their thresholds (<=, >=).",
        "Mid-write tolerance: a malformed FINAL JSONL line is skipped with a "
        "warning; an incomplete attempt-1 row set (missing gated-arm row or "
        "sidecar integrity failure) falls back to attempt 0 with a provisional "
        "note.  Interior malformed lines and duplicate keys abort.",
        "G_c left PENDING: the V3-5 rescore artifact "
        "outputs/sweeps/gold_rescore_p4.jsonl is required and absent; ALT1_GO "
        "is reported as a skeleton with the R_c>=5 candidate set only.",
        "EP enters b_c only when its row has converged == true (registered "
        "'converged EP'); otherwise b_c narrows to VI-only with a warning.",
        "q_P4 used at full float precision (3.828615828610777), asserted to "
        "round to the registered 3.8286158286.",
    ]

    cells = []
    for c in range(6):
        cells.append(analyze_cell(c, chosen[c], idx, archived, vi_rows, ep_rows,
                                  q, warnings))

    # PENDING_REMEDIATION applies only where remediation is REGISTERED: the
    # 2026-07-18 ruling invoked the single predeclared remediation event for
    # cells failing the V4-2/acceptance DIAGNOSTIC gates at attempt 0 (in
    # fact adult/0.3 only) -- i.e. a cell evaluated at attempt 0 whose
    # diagnostics fail while its attempt-1 gated rows have not landed yet.
    # Equivalence/precision failures are plain FAIL (the remediation event
    # is spent; the v2E kill criterion would then apply).
    att1_missing = [c for c in range(6) if chosen[c] == 0
                    and not cells[c]["diagnostics_pass"]
                    and not all((c, a, 1) in idx for a in GATED_ARMS)]
    for c in att1_missing:
        cells[c]["status"] = ("PENDING_REMEDIATION (attempt-0 diagnostics fail; "
                              "attempt-1 rows not landed at analysis time)")
    for c in range(6):
        cells[c].setdefault("status", "PASS" if cells[c]["cell_pass"] else "FAIL")

    p4_all6 = all(cl["cell_pass"] for cl in cells)
    p4_final = not att1_missing
    r_candidates = [cl["cell_idx"] for cl in cells if np.isfinite(cl["s2b"]["R_c"])
                    and cl["s2b"]["R_c"] >= 5.0]
    rescore_exists = RESCORE_PATH.exists()
    if rescore_exists:
        warnings.append(f"{RESCORE_PATH} exists but G_c scoring is deferred to the "
                        "S2b rescore analysis (this report leaves G_c pending)")
    # ALT1_GO = P4_ALL6_STABLE AND |{c: R_c >= 5 AND G_c < 1}| >= 3.  G_c is
    # pending, but the boolean is already decidable when the R_c clause alone
    # caps the conjunction set below 3 (G_c cannot enlarge it), or when
    # P4_ALL6_STABLE is finally false.
    if p4_final and len(r_candidates) < 3:
        alt1_go = False
        alt1_note = (f"decidable without G_c: R_c >= 5 holds on "
                     f"{len(r_candidates)}/6 cells (< 3 required); the pending "
                     "G_c clause cannot enlarge the same-cell conjunction set")
    elif p4_final and not p4_all6:
        alt1_go = False
        alt1_note = "P4_ALL6_STABLE is finally false"
    else:
        alt1_go = None
        alt1_note = ("PENDING: G_c requires the V3-5 rescore "
                     "(gold_rescore_p4.jsonl), which has not run"
                     + ("" if p4_final else "; P4_ALL6_STABLE also non-final"))

    report = {
        "schema": "p4-gate-report-1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "precommit": "campaigns/screens_precommit.md#V3-1+V3-3+V3-4"
                     "+V4-1+V4-2+amendment-v2C/E+remediation-ruling-2026-07-18",
        "script": {"path": "scripts/p4_gate_analysis.py",
                   "sha256": sha256_file(Path(__file__))},
        "inputs": {
            "gold_p4": {"path": str(args.gold), "sha256": sha256_file(args.gold),
                        "rows": len(rows)},
            "archived_gold": {"path": str(args.archived_gold),
                              "sha256": sha256_file(args.archived_gold)},
            "vi_baseline": {"path": str(args.vi), "sha256": sha256_file(args.vi)},
            "rr_extrapolation_script_sha256": rr_sha,
            "gold_rescore_present": rescore_exists,
        },
        "constants": {"q_p4": q, "d_p4": D_P4, "rhat_max": RHAT_MAX,
                      "ess_min": ESS_MIN, "acceptance_band": [ACC_LO, ACC_HI],
                      "precision_ceiling": PREC_CEIL, "equivalence_ceiling": EQUIV_CEIL,
                      "batch_len": BATCH_LEN, "n_seeds": N_SEEDS},
        "attempt_selection": {"mode": "auto" if attempt_map is None else "explicit",
                              "chosen": {str(c): chosen[c] for c in range(6)},
                              "notes": notes},
        "judgment_calls": judgment_calls,
        "warnings": warnings,
        "cells": cells,
        "booleans": {
            "P4_ALL6_STABLE": p4_all6,
            "P4_ALL6_STABLE_final": p4_final,
            "P4_ALL6_STABLE_note": None if p4_final else
                "non-final: predeclared remediation invoked (2026-07-18 ruling); "
                f"attempt-1 rows for cell(s) {att1_missing} not landed at analysis time",
            "ALT1_GO": alt1_go,
            "ALT1_GO_note": "ALT1_GO := P4_ALL6_STABLE AND >=3 same cells with "
                            f"(R_c >= 5 AND G_c < 1).  {alt1_note}",
            "R_ge_5_cells": r_candidates,
            "G_c_status": "PENDING: outputs/sweeps/gold_rescore_p4.jsonl absent "
                          "(V3-5 rescore not yet run)" if not rescore_exists else
                          "rescore file present; G_c scoring deferred to the S2b "
                          "rescore analysis",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, sort_keys=True)
        f.write("\n")

    print_console(report)
    print(f"\nreport written: {args.report}")
    return report


def print_console(rep):
    b = rep["booleans"]
    print("\n=== P4 gate report ({}) ===".format(rep["generated_utc"]))
    print(f"q_P4={rep['constants']['q_p4']:.10f}  D={rep['constants']['d_p4']}  "
          f"gates: Rhat<=1.01 ESS>=400 acc[0.45,0.80] MCSEc/SD<=0.02 "
          f"|D|+q*MCSEc<=0.10*SD")
    print("SD convention (gate-bearing): sd_draws_est = sqrt(K)*BesselSD(series) "
          "-- see judgment_calls[0]; series-SD sensitivity in the JSON")
    hdr = (f"{'cell':<14}{'att':>3} {'diagZ':>6}{'diagC':>6}{'accZ':>5}{'accC':>5}"
           f"{'sPrec':>6}{'sEq':>5}{'xPrec':>6}{'xEq':>5} {'PASS':>5}  "
           f"{'sEqM':>6}{'xEqM':>6} {'r95':>9} {'shift_sq':>9} {'spdR':>6} "
           f"{'b_c':>9} {'R_c':>9}")
    print(hdr)
    print("-" * len(hdr))
    fm = lambda v: "ok" if v else "FAIL"
    for c in rep["cells"]:
        g = c["gates"]
        fams = c["families"]
        print(f"{c['dataset'] + '/' + str(c['prior']):<14}{c['attempt_used']:>3} "
              f"{fm(g['diag_zero_ok']):>5}{fm(g['diag_chol_ok']):>6}"
              f"{fm(g['acc_zero_ok']):>5}{fm(g['acc_chol_ok']):>5}"
              f"{fm(g['split_precision_ok']):>6}{fm(g['split_equivalence_ok']):>5}"
              f"{fm(g['cross_precision_ok']):>6}{fm(g['cross_equivalence_ok']):>5} "
              f"{fm(c['cell_pass']):>5}  "
              f"{fams['split_half']['max_equiv_margin']:>6.3f}"
              f"{fams['cross_arm']['max_equiv_margin']:>6.3f} "
              f"{c['r95']:>9.3e} {c['shift_sq']:>9.3e} "
              f"{c['spread_ratio_chol_over_archived_descriptive']:>6.2f} "
              f"{c['s2b']['b_c']:>9.3e} {c['s2b']['R_c']:>9.3e}")
        if c["status"] != ("PASS" if c["cell_pass"] else "FAIL"):
            print(f"    -> {c['status']}")
    print(f"\nP4_ALL6_STABLE = {b['P4_ALL6_STABLE']} "
          f"(final: {b['P4_ALL6_STABLE_final']})")
    if b["P4_ALL6_STABLE_note"]:
        print(f"  note: {b['P4_ALL6_STABLE_note']}")
    print(f"ALT1_GO = {b['ALT1_GO']} -- {b['ALT1_GO_note']}")
    print(f"R_c >= 5 candidate cells: {b['R_ge_5_cells']}")
    if rep["warnings"]:
        print("\nwarnings:")
        for w in rep["warnings"]:
            print(f"  - {w}")


# --- selftest (synthetic fixtures only; touches no accepted artifact) --------

def _extract_func_source(path, names):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    return out


def selftest():
    _import_diag_deps()
    n_pass, n_fail = [0], [0]

    def check(name, cond, detail=""):
        ok = bool(cond)
        (n_pass if ok else n_fail)[0] += 1
        print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    print("P4 gate-analysis selftest: synthetic fixtures only; no accepted "
          "artifact is read.")

    print("[1] anti-drift: ported estimators byte-match scripts/rr_extrapolation.py")
    theirs = _extract_func_source(RR_SCRIPT, PORTED_FUNCS)
    ours = _extract_func_source(Path(__file__), PORTED_FUNCS)
    for fn in PORTED_FUNCS:
        check(f"{fn} source identical", theirs.get(fn) == ours.get(fn))

    print("[2] q_P4 constant")
    q = q_p4()
    check("q_P4 rounds to registered 3.8286158286", round(q, 10) == Q_P4_REGISTERED,
          f"computed {q!r}")

    print("[3] MCSE from batch means (V3-1)")
    bm = np.array([[1.0], [2.0], [3.0], [4.0]])
    check("deterministic: sd([1,2,3,4],ddof=1)/sqrt(4)",
          abs(float(p4_mcse(bm)[0]) - math.sqrt(5.0 / 3.0) / 2.0) <= 1e-15)
    rs = np.random.RandomState(990_100_000)
    bm90 = rs.standard_normal((90, 3))
    check("iid N(0,1) 90 batches: mcse*sqrt(90) within 25% of 1",
          bool(np.all(np.abs(p4_mcse(bm90) * math.sqrt(90) - 1.0) <= 0.25)))
    check("combined MCSE 3-4-5", abs(float(combined_mcse([3e-3], [4e-3])[0]) - 5e-3) <= 1e-18)

    print("[4] split-half construction (V3-4)")
    bm1 = np.array([[[1.0], [2.0], [3.0], [4.0]]])  # 1 seed, 4 batches
    d, ma, mb2, mu_a, mu_b = split_half_family(bm1)
    check("halves align with batch boundaries: means 1.5/3.5, delta -2",
          abs(mu_a[0] - 1.5) <= 1e-15 and abs(mu_b[0] - 3.5) <= 1e-15
          and abs(d[0] + 2.0) <= 1e-15)
    check("per-half MCSE = sd(half,ddof=1)/sqrt(2)",
          abs(float(ma[0]) - np.std([1.0, 2.0], ddof=1) / math.sqrt(2)) <= 1e-15)

    print("[5] V4-1 gates at the 0.0766 + 0.0234 boundary (SD=1, MCSE_c=0.02)")
    sd1 = np.array([1.0])
    mc = np.array([0.02])
    check("q*MCSE_c = 0.0766 (4dp)", round(q * 0.02, 4) == 0.0766, f"{q * 0.02:.6f}")
    check("precision passes at ratio == 0.02 (inclusive)", precision_gate(mc, sd1)[0])
    check("precision fails just above 0.02",
          not precision_gate(np.array([0.02000001]), sd1)[0])
    check("equivalence passes at |Delta|=0.0234 (0.0234+0.0766 <= 0.10)",
          equivalence_gate(np.array([0.0234]), mc, sd1, q)[0],
          f"lhs={0.0234 + q * 0.02:.6f}")
    check("equivalence fails at |Delta|=0.0235 (0.0235+0.0766 > 0.10)",
          not equivalence_gate(np.array([0.0235]), mc, sd1, q)[0])
    check("descriptive z-count: |Delta|=0.0234 < q*MCSE_c not counted",
          z_nonreject_descriptive(np.array([0.0234]), mc, q) == 0)
    check("nonfinite SD fails equivalence",
          not equivalence_gate(np.array([0.0]), mc, np.array([np.nan]), q)[0])

    print("[6] acceptance gate [0.45, 0.80] inclusive")
    check("0.45/0.80 bounds pass", acceptance_ok([0.45, 0.60, 0.80]))
    check("0.449 fails", not acceptance_ok([0.449, 0.6, 0.7]))
    check("0.801 fails", not acceptance_ok([0.6, 0.801, 0.7]))
    check("nan fails", not acceptance_ok([0.6, float("nan"), 0.7]))

    print("[7] r95 / R_c / shift_sq formulas")
    m = np.full(4, 2.0e-3)
    check("r95 = q*m*sqrt(d) on constant MCSE",
          abs(r95_from_mcse(m, q) - q * 2.0e-3 * 2.0) <= 1e-15)
    check("R_c = 0 at b <= r95", s2b_r(1.0, 1.0) == 0.0 and s2b_r(0.5, 1.0) == 0.0)
    check("R_c = 25 at b = 6*r95", abs(s2b_r(6.0, 1.0) - 25.0) <= 1e-12)
    check("shift_sq = ||[0.3,0.4]||^2 / 0.5 = 0.5",
          abs(shift_sq_stat([0.3, 0.4], [0.0, 0.0], 0.5) - 0.5) <= 1e-15)

    print("[8] V4-2 estimators: iid, shifted-chains, AR(1) known ESS")
    rs2 = np.random.RandomState(990_200_000)
    iid = rs2.standard_normal((3, 4000, 2))
    di = diagnostics(iid)
    s_total = 3 * 4000
    check("iid: max Rhat <= 1.005", max(di["split_rhat"]) <= 1.005,
          f"max={max(di['split_rhat']):.4f}")
    check("iid: bulk ESS in [0.8, 1.25]*S",
          0.8 * s_total <= min(di["bulk_ess"]) and max(di["bulk_ess"]) <= 1.25 * s_total,
          f"min={min(di['bulk_ess']):.0f}, S={s_total}")
    shifted = iid.copy() + np.linspace(-1.0, 1.0, 3)[:, None, None]
    ds_ = diagnostics(shifted)
    check("shifted chains: Rhat > 1.01 (gate would fail)",
          max(ds_["split_rhat"]) > RHAT_MAX, f"max={max(ds_['split_rhat']):.3f}")
    phi = 0.999
    innov = rs2.standard_normal((3, 20000, 1)) * math.sqrt(1 - phi ** 2)
    ar1 = np.empty_like(innov)
    state = rs2.standard_normal((3, 1))
    for t in range(20000):
        state = phi * state + innov[:, t, :]
        ar1[:, t, :] = state
    da = diagnostics(ar1)
    ess_true = 3 * 20000 * (1 - phi) / (1 + phi)
    ratio = da["bulk_ess"][0] / ess_true
    check("AR(1) phi=0.999: ESS < 400 (gate would fail)",
          min(da["bulk_ess"]) < ESS_MIN, f"ess={min(da['bulk_ess']):.0f}")
    check("AR(1) ESS within [0.3, 3]x known value (V4-2 selftest contract)",
          0.3 <= ratio <= 3.0, f"est={da['bulk_ess'][0]:.0f} true={ess_true:.0f} "
          f"ratio={ratio:.2f}")

    print("[9] loader: duplicate abort, schema check, mid-write tolerance")
    base = {"schema_version": P4_SCHEMA_VERSION, "cell_idx": 0, "arm_idx": 0,
            "attempt": 0, "dataset": "wdbc", "prior": 0.1, "gold_id": "t"}
    try:
        index_p4_rows([base, dict(base)])
        check("duplicate (cell,arm,attempt) aborts", False)
    except SystemExit:
        check("duplicate (cell,arm,attempt) aborts", True)
    try:
        index_p4_rows([dict(base, schema_version="bogus")])
        check("wrong schema_version aborts", False)
    except SystemExit:
        check("wrong schema_version aborts", True)
    try:
        index_p4_rows([dict(base, dataset="adult")])
        check("cell_idx/dataset mismatch aborts", False)
    except SystemExit:
        check("cell_idx/dataset mismatch aborts", True)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.jsonl"
        p.write_text(json.dumps(base) + "\n" + '{"partial', encoding="utf-8")
        w = []
        rows = load_jsonl_tolerant(p, w)
        check("malformed FINAL line skipped with warning",
              len(rows) == 1 and len(w) == 1)
        p.write_text('{"partial\n' + json.dumps(base) + "\n", encoding="utf-8")
        try:
            load_jsonl_tolerant(p, [])
            check("malformed INTERIOR line aborts", False)
        except SystemExit:
            check("malformed INTERIOR line aborts", True)

    print("[10] attempt selection (auto prefers complete attempt 1)")
    def mkrow(c, a, t):
        return dict(base, cell_idx=c, arm_idx=a, attempt=t,
                    dataset=CELLS[c][0], prior=CELLS[c][1], gold_id=f"t{c}{a}{t}")
    idx0 = index_p4_rows([mkrow(0, a, 0) for a in (0, 1, 2)])
    notes = []
    check("attempt-0-only cell -> 0",
          select_attempts(idx0, None, lambda r: True, notes, cells=[0])[0] == 0)
    idx1 = index_p4_rows([mkrow(0, a, t) for a in (0, 1, 2) for t in (0, 1)])
    check("complete attempt-1 -> 1",
          select_attempts(idx1, None, lambda r: True, [], cells=[0])[0] == 1)
    idxp = index_p4_rows([mkrow(0, a, 0) for a in (0, 1, 2)] + [mkrow(0, 0, 1)])
    notes = []
    check("partial attempt-1 (chol missing) -> fallback 0 + provisional note",
          select_attempts(idxp, None, lambda r: True, notes, cells=[0])[0] == 0
          and len(notes) == 1)
    notes = []
    check("attempt-1 sidecar integrity failure -> fallback 0 + note",
          select_attempts(idx1, None, lambda r: r["attempt"] == 0, notes,
                          cells=[0])[0] == 0
          and len(notes) == 1)
    check("explicit --attempt-map wins",
          select_attempts(idx1, {0: 0}, lambda r: True, [], cells=[0])[0] == 0)
    check("attempt-map parse", parse_attempt_map("5:1,0:0") == {5: 1, 0: 0}
          and parse_attempt_map("auto") is None)

    print("[11] SD_draws estimator: sqrt(K)*BesselSD(ensemble-mean series)")
    rs3 = np.random.RandomState(990_300_000)
    sigma_true, k_ch = 2.5, 16
    chain_draws = sigma_true * rs3.standard_normal((k_ch, 30000, 1))
    series_mean = chain_draws.mean(axis=0)  # (30000, 1) ensemble-mean series
    sd_est = math.sqrt(k_ch) * series_mean.std(axis=0, ddof=1)
    check("sqrt(K)*SD(series) recovers the chain-draw SD within 2% "
          "(iid chains fixture)", abs(float(sd_est[0]) / sigma_true - 1.0) <= 0.02,
          f"est={float(sd_est[0]):.4f} true={sigma_true}")
    sd_direct = chain_draws.reshape(-1, 1).std(axis=0, ddof=1)
    check("direct pooled chain-draw SD agrees with the estimator within 2%",
          abs(float(sd_est[0]) / float(sd_direct[0]) - 1.0) <= 0.02)

    total = n_pass[0] + n_fail[0]
    print(f"\nselftest: {n_pass[0]}/{total} PASS" +
          ("" if n_fail[0] == 0 else f"  ({n_fail[0]} FAILED)"))
    return n_fail[0] == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", type=Path, default=GOLD_DEFAULT)
    ap.add_argument("--archived-gold", type=Path, default=ARCHIVED_GOLD_DEFAULT)
    ap.add_argument("--vi", type=Path, default=VI_DEFAULT)
    ap.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--attempt-map", default="auto",
                    help="'auto' (prefer complete attempt-1 per cell) or "
                         "'cell:att,...' e.g. '5:1'")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    run_analysis(args)


if __name__ == "__main__":
    main()
