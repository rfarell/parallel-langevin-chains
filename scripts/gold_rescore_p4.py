"""V3-5 ordering rescore: rerun the four archived gold-endtask table arms and
score the SAME retained per-ensemble estimates against BOTH golds.

Binding protocol: campaigns/screens_precommit.md, Amendment v3
V3-5 (paired rescoring) + Amendment v4 V4-5 (two-row schema, loader aborts) +
Amendment v2 section E (ordering-change rule) + the 2026-07-18 remediation
ruling (hardened gold = passing attempt per cell: attempt 0 everywhere except
adult/0.3, whose successful remediation attempt 1 replaces attempt 0).

The four arms are exactly the archived table rows of
paper/tables/gold_endtask.tex (scripts/rr_extrapolation.py estimator_specs,
primary pair + context; K=8 chains, matched budgets, zero starts, 50%
burn-in):
  arm_row 0  iid  ULA @ eta1 = 4*eta_c        (table "iid ULA")
  arm_row 1  C*   ULA @ eta1 = 4*eta_c        (table "C* ULA")
  arm_row 2  C*   ULA @ eta2 = 2*eta_c        (table "C* ULA (fine)")
  arm_row 3  C*   MALA @ archived gold eta    (table "MALA", context)
with the archived per-dataset settings (n_ens 128/128/64, budget 2400/2400/
1800 steps, eta_c = min(0.03/(prior+0.25), 0.08)).  Dynamics code is IMPORTED
from scripts/rr_extrapolation.py (ula_time_avg / mala_time_avg / warm_up /
make_noise / load_dataset), so the rerun uses the archived implementations
verbatim; only the seeds are fresh, from the registered 344M map (V3-3):
  seed = 344,000,000 + cell_idx*100,000 + arm_row*10,000 + seed_idx*1,000
(cell_idx per V3-3; seed_idx in {0,1,2} mirrors rep0..2).

Output (outputs/sweeps/gold_rescore_p4.jsonl, schema_version "p4-1"):
  kind=rescore_fit    one per (cell, arm, seed_idx); RETAINS the per-ensemble
                      estimator means (theta_ens, float64, never discarded);
  kind=rescore_score  exactly TWO per source_fit_id, keyed
                      (source_fit_id, score_gold_id): the same theta_ens
                      scored against the archived gold mean and against the
                      hardened gold mean (paired contrast; scoring is
                      host-side float64 numpy, identical formulas to
                      run_estimator: per-ensemble squared error, median/mean,
                      bias^2, trace variance, finite-ensemble filter);
  kind=rescore_run_meta  one per invocation (argv, GPU preflight, gold shas).
load_rescore() aborts on any duplicate key, on a source_fit_id with a score
row count other than two, on a wrong schema_version, and (when the expected
map is given) on any unexpected score_gold_id.

--selftest is CPU-only (numpy shim from rr_extrapolation; cupy is never
imported).  The accepted run is GPU-gated: section H preflight (nvidia-smi
compute-apps on the selected GPU UUID, fail-closed) runs before cupy import,
and the hardened-gold selection aborts while adult/0.3's remediation rows
have not landed.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rr_extrapolation as rr  # noqa: E402  (archived dynamics, verbatim)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "p4-1"
PRECOMMIT = ("campaigns/screens_precommit.md"
             "#V3-5+V4-5+amendment-2026-07-18-remediation-ruling")
RESCORE_SEED_BASE = 344_000_000
CHAINS = 8  # archived: run_estimator was always called with chains=8
ARM_NAMES = {0: "iid_eta1", 1: "zs_eta1", 2: "zs_eta2", 3: "zs_mala"}
# 2026-07-18 ruling: attempt 0 for the five passing cells, attempt 1 for
# adult/0.3 (cell_idx 5) once its remediation rows land.
PASSING_ATTEMPT_DEFAULT = [0, 0, 0, 0, 0, 1]
ARCHIVED_GOLD_SHA256 = "b376b814f282cdddadf0890031306f8fa376fafeb0405908b31d11f314cc55fe"
HARDENED_SCHEMA_VERSION = "p4-gold-1"
OUT_DEFAULT = Path("outputs/sweeps/gold_rescore_p4.jsonl")
ARCHIVED_GOLD_DEFAULT = Path("outputs/sweeps/rr_extrapolation_gold.jsonl")
HARDENED_GOLD_DEFAULT = Path("outputs/sweeps/rr_extrapolation_gold_p4_fd354ef.jsonl")


def rescore_seed(cell_idx, arm_row, seed_idx):
    """Registered 344M map (V3-3, sampler-column reruns)."""
    assert 0 <= cell_idx <= 5 and 0 <= arm_row <= 3 and 0 <= seed_idx <= 2
    return RESCORE_SEED_BASE + cell_idx * 100_000 + arm_row * 10_000 + seed_idx * 1_000


def all_rescore_seeds():
    return [rescore_seed(c, a, s) for c in range(6) for a in range(4) for s in range(3)]


def build_arm_specs(eta_c, budget):
    """The four archived table arms, pulled from rr.estimator_specs itself so
    the rerun spec cannot drift from the archived generator."""
    specs = rr.estimator_specs(eta_c, budget)

    def pick(**kw):
        hits = [s for s in specs if all(s.get(k) == v for k, v in kw.items())]
        if len(hits) != 1:
            raise SystemExit(f"archived spec lookup {kw} matched {len(hits)} rows")
        return hits[0]

    return [
        (0, ARM_NAMES[0], pick(estimator="single", pair="primary", coupling="iid", level="eta1")),
        (1, ARM_NAMES[1], pick(estimator="single", pair="primary", coupling="zero_sum", level="eta1")),
        (2, ARM_NAMES[2], pick(estimator="single", pair="primary", coupling="zero_sum", level="eta2")),
        (3, ARM_NAMES[3], pick(estimator="mala", pair="context")),
    ]


def run_arm(spec, seed, arch_mala_eta, n_ens, chains, dim, prior, data_x, data_y, chunk_rows):
    """One rerun arm with the archived dynamics functions.  Mirrors
    run_estimator's dispatch exactly (eta1 None -> archived gold mala_eta;
    warm-up excluded from timing).  Returns host float64 per-ensemble means."""
    eta1 = spec["eta1"] if spec["eta1"] is not None else arch_mala_eta
    rr.warm_up(spec["kernel"], spec["coupling"], eta1, chains, dim, prior,
               data_x, data_y, chunk_rows)
    t0 = time.perf_counter()
    accept_rate = None
    if spec["estimator"] == "single":
        theta = rr.ula_time_avg(seed, spec["coupling"], eta1, spec["steps1"], n_ens,
                                chains, dim, prior, data_x, data_y, chunk_rows)
    else:  # mala context
        theta, accept_rate = rr.mala_time_avg(seed, spec["coupling"], eta1, spec["steps1"],
                                              n_ens, chains, dim, prior, data_x, data_y,
                                              chunk_rows)
    rr.cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    return np.asarray(theta.get(), dtype=np.float64), accept_rate, eta1, elapsed


def score_against(theta_ens, gold_mean):
    """run_estimator's scoring formulas, host-side float64 numpy.  The finite
    filter depends on theta only, so it is identical for both references."""
    theta_ens = np.asarray(theta_ens, dtype=np.float64)
    gold = np.asarray(gold_mean, dtype=np.float64)
    err = theta_ens - gold[None, :]
    sq = np.sum(err * err, axis=1)
    finite = np.isfinite(sq)
    n_finite = int(finite.sum())
    if n_finite >= 2:
        sq_f = sq[finite]
        theta_f = theta_ens[finite]
        theta_bar = theta_f.mean(axis=0)
        return {
            "n_finite": n_finite,
            "mse_gold_median": float(np.median(sq_f)),
            "mse_gold_mean": float(sq_f.mean()),
            "bias_sq_gold": float(np.sum((theta_bar - gold) ** 2)),
            "trace_var": float(np.mean(np.sum((theta_f - theta_bar[None, :]) ** 2, axis=1))),
        }
    return {"n_finite": n_finite, "mse_gold_median": None, "mse_gold_mean": None,
            "bias_sq_gold": None, "trace_var": None}


def source_fit_id(ds, prior, arm_row, arm_name, seed_idx, seed):
    return f"p4rescore|{ds}|prior={prior}|arm={arm_name}|arm_row={arm_row}|rep={seed_idx}|seed={seed}"


# ----------------------------------------------------------------------------
# Gold loading / selection (fail-closed).
# ----------------------------------------------------------------------------

def load_archived_gold(path, expected_sha=ARCHIVED_GOLD_SHA256):
    """Archived gold rows keyed by cell; the file content is pinned by sha256
    (the archived rows carry no schema_version).  Duplicate cells abort."""
    if expected_sha is not None:
        actual = rr.sha256_file(path)
        if actual != expected_sha:
            raise SystemExit(f"archived gold {path}: sha256 {actual} != pinned {expected_sha} "
                             "(refusing to score against an unpinned reference)")
    cells = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "gold":
            continue
        key = (row["dataset"], row["prior"])
        if key in cells:
            raise SystemExit(f"archived gold {path}: duplicate row for cell {key}")
        cells[key] = row
    return cells


def select_hardened_gold(path, cells_needed, passing_attempts):
    """Hardened-gold selection per V3-1 + the 2026-07-18 ruling: for each cell
    the chol (primary dispersed) arm row at that cell's PASSING attempt.
    Aborts on wrong schema_version, duplicates, or missing rows (fail-closed
    while adult/0.3's remediation rows have not landed)."""
    found = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "gold":
            continue
        sv = row.get("schema_version")
        if sv != HARDENED_SCHEMA_VERSION:
            raise SystemExit(f"hardened gold {path}: schema_version {sv!r} != "
                             f"{HARDENED_SCHEMA_VERSION!r}")
        if row.get("start_mode") != "chol":
            continue
        cell = (row["dataset"], row["prior"])
        if cell not in cells_needed:
            continue
        cidx = rr.p4_cell_idx(*cell)
        if row.get("attempt") != passing_attempts[cidx]:
            continue
        if row.get("arm_idx") != rr.P4_ARM_IDX["chol"]:
            raise SystemExit(f"hardened gold {path}: chol row for {cell} carries "
                             f"arm_idx {row.get('arm_idx')!r} != {rr.P4_ARM_IDX['chol']}")
        if cell in found:
            raise SystemExit(f"hardened gold {path}: duplicate chol/attempt="
                             f"{passing_attempts[cidx]} row for cell {cell}")
        found[cell] = row
    missing = [c for c in cells_needed if c not in found]
    if missing:
        raise SystemExit(
            f"hardened gold {path}: no chol row at the registered passing attempt for "
            f"cells {missing} (attempts map {passing_attempts}); if this is adult/0.3 "
            "the remediation rows have not landed yet -- the rescore stays blocked")
    return found


# ----------------------------------------------------------------------------
# Output cache (fail-closed resume) and the strict loader (V4-5).
# ----------------------------------------------------------------------------

def load_cache(path):
    """source_fit_ids whose fit + both score rows are already on disk.
    Any partial or duplicated triple aborts (manual repair, never last-win)."""
    if not Path(path).exists():
        return set()
    fit_counts = defaultdict(int)
    score_keys = defaultdict(int)
    scores_by_fit = defaultdict(set)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.get("kind")
        if kind == "rescore_fit":
            fit_counts[row["source_fit_id"]] += 1
        elif kind == "rescore_score":
            key = (row["source_fit_id"], row["score_gold_id"])
            score_keys[key] += 1
            scores_by_fit[row["source_fit_id"]].add(row["score_gold_id"])
    for key, n in score_keys.items():
        if n > 1:
            raise SystemExit(f"{path}: duplicate score row {key} (refusing to resume)")
    for sfid, n in fit_counts.items():
        if n > 1:
            raise SystemExit(f"{path}: duplicate fit row {sfid} (refusing to resume)")
        if len(scores_by_fit.get(sfid, ())) != 2:
            raise SystemExit(f"{path}: fit {sfid} has {len(scores_by_fit.get(sfid, ()))} "
                             "score rows (expected 2); repair the file before resuming")
    for sfid in scores_by_fit:
        if sfid not in fit_counts:
            raise SystemExit(f"{path}: score rows for unknown fit {sfid}")
    return set(fit_counts)


def load_rescore(path, expected_schema_version=SCHEMA_VERSION, expected_gold_ids=None):
    """Strict analysis loader (V3-5 + V4-5).

    expected_gold_ids: optional {(dataset, prior): {"archived": gid,
    "hardened": gid}}.  Aborts on: wrong/missing schema_version on ANY row,
    duplicate source_fit_id, duplicate (source_fit_id, score_gold_id), a
    source_fit_id whose score-row count is not exactly two, score rows without
    a fit row, and (when the map is given) any cell or score_gold_id outside
    the expected map.  Returns (fits, scores) with
    fits = {source_fit_id: row}, scores = {source_fit_id: {score_gold_id: row}}."""
    fits = {}
    scores = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sv = row.get("schema_version")
        if sv != expected_schema_version:
            raise SystemExit(f"rescore loader: {path}: schema_version {sv!r} != "
                             f"expected {expected_schema_version!r}")
        kind = row.get("kind")
        if kind == "rescore_fit":
            sfid = row["source_fit_id"]
            if sfid in fits:
                raise SystemExit(f"rescore loader: {path}: duplicate fit row {sfid!r}")
            fits[sfid] = row
        elif kind == "rescore_score":
            sfid, sgid = row["source_fit_id"], row["score_gold_id"]
            per = scores.setdefault(sfid, {})
            if sgid in per:
                raise SystemExit(f"rescore loader: {path}: duplicate score row "
                                 f"({sfid!r}, {sgid!r})")
            per[sgid] = row
        # other kinds (rescore_run_meta) are schema-checked and passed over
    for sfid in scores:
        if sfid not in fits:
            raise SystemExit(f"rescore loader: {path}: score rows for unknown fit {sfid!r}")
    for sfid, fit in fits.items():
        got = scores.get(sfid, {})
        if len(got) != 2:
            raise SystemExit(f"rescore loader: {path}: fit {sfid!r} has {len(got)} score "
                             "rows (exactly two required)")
        if expected_gold_ids is not None:
            cell = (fit["dataset"], fit["prior"])
            if cell not in expected_gold_ids:
                raise SystemExit(f"rescore loader: {path}: fit {sfid!r} cell {cell} not in "
                                 "the expected gold-id map")
            exp = expected_gold_ids[cell]
            want = {exp["archived"], exp["hardened"]}
            if set(got) != want:
                raise SystemExit(f"rescore loader: {path}: fit {sfid!r} score_gold_ids "
                                 f"{sorted(got)} != expected {sorted(want)}")
    return fits, scores


# ----------------------------------------------------------------------------
# GPU preflight (precommit section H, fail-closed).
# ----------------------------------------------------------------------------

def _preflight_decide(gpu_csv, apps_csv, gpu_index):
    """Pure decision: (ok, reason).  ABORT if any compute app sits on the
    selected GPU's UUID or anything is unparseable (ownership unclear)."""
    gmap = {}
    for ln in gpu_csv.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or not parts[1].startswith("GPU-"):
            return False, f"unparseable gpu line {ln!r}"
        gmap[int(parts[0])] = parts[1]
    if not gmap:
        return False, "no GPUs reported"
    if gpu_index not in gmap:
        return False, f"gpu index {gpu_index} not present in {sorted(gmap)}"
    uuid = gmap[gpu_index]
    for ln in apps_csv.splitlines():
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3 or not parts[0].startswith("GPU-"):
            return False, f"unparseable compute-apps line {ln!r}"
        if parts[0] == uuid:
            return False, f"compute process present on selected GPU {uuid}: {ln!r}"
    return True, uuid


def gpu_preflight(gpu_index):
    try:
        gpu_csv = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
        apps_csv = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
             "--format=csv,noheader"], text=True)
    except Exception as exc:  # noqa: BLE001 - fail closed on any preflight error
        sys.exit(f"GPU preflight failed to run nvidia-smi: {exc}")
    ok, reason = _preflight_decide(gpu_csv, apps_csv, gpu_index)
    text = (f"$ nvidia-smi --query-gpu=index,uuid --format=csv,noheader\n{gpu_csv}"
            f"$ nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name "
            f"--format=csv,noheader\n{apps_csv}")
    if not ok:
        sys.exit(f"GPU preflight ABORT (section H): {reason}\n{text}")
    return text, reason  # reason == selected uuid on success


# ----------------------------------------------------------------------------
# Core: rescore one cell (shared by --run and --selftest).
# ----------------------------------------------------------------------------

def append_row(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()


def rescore_cell(out_path, ds, prior, cell_idx, arch_row, hard_row, data_x, data_y,
                 n_ens, chains, budget, chunk_rows, cache, common_prov, seed_idxs=(0, 1, 2)):
    dim = int(data_x.shape[1])
    for label, row in (("archived", arch_row), ("hardened", hard_row)):
        if row["dim"] != dim:
            raise SystemExit(f"{label} gold dim {row['dim']} != data dim {dim} for "
                             f"({ds}, {prior})")
    if arch_row["mala_eta"] != hard_row["mala_eta"]:
        raise SystemExit(f"frozen-eta violation for ({ds}, {prior}): archived mala_eta "
                         f"{arch_row['mala_eta']!r} != hardened {hard_row['mala_eta']!r}")
    eta_c = rr.default_eta(prior)
    arch_gold = np.asarray(arch_row["gold_mean"], dtype=np.float64)
    hard_gold = np.asarray(hard_row["gold_mean"], dtype=np.float64)
    n_new = 0
    for arm_row, arm_name, spec in build_arm_specs(eta_c, budget):
        for j in seed_idxs:
            seed = rescore_seed(cell_idx, arm_row, j)
            sfid = source_fit_id(ds, prior, arm_row, arm_name, j, seed)
            if sfid in cache:
                print(f"rescore cache hit: {sfid}", flush=True)
                continue
            theta, accept_rate, eta1, elapsed = run_arm(
                spec, seed, arch_row["mala_eta"], n_ens, chains, dim, prior,
                data_x, data_y, chunk_rows)
            fit = {
                "kind": "rescore_fit",
                "schema_version": SCHEMA_VERSION,
                "target": "gold_rescore_p4",
                "precommit": PRECOMMIT,
                "source_fit_id": sfid,
                "dataset": ds,
                "prior": prior,
                "dim": dim,
                "cell_idx": cell_idx,
                "arm_row": arm_row,
                "arm_name": arm_name,
                "estimator": spec["estimator"],
                "pair": spec["pair"],
                "coupling": spec["coupling"],
                "level": spec["level"],
                "kernel": spec["kernel"],
                "eta1": eta1,
                "eta2": spec["eta2"],
                "steps1": spec["steps1"],
                "steps2": spec["steps2"],
                "budget": budget,
                "eta_c": eta_c,
                "accept_rate": accept_rate,
                "seed": seed,
                "seed_idx": j,
                "replicate": j,
                "n_ens": n_ens,
                "chains": chains,
                "theta_ens": [[float(v) for v in row_] for row_ in theta],
                "elapsed_sec": elapsed,
                "elapsed_sec_per_ensemble": elapsed / n_ens,
                **common_prov,
            }
            append_row(out_path, fit)
            for gold_kind, gold_row, gold_vec in (
                    ("archived", arch_row, arch_gold), ("hardened", hard_row, hard_gold)):
                sc = score_against(theta, gold_vec)
                score = {
                    "kind": "rescore_score",
                    "schema_version": SCHEMA_VERSION,
                    "target": "gold_rescore_p4",
                    "precommit": PRECOMMIT,
                    "source_fit_id": sfid,
                    "score_gold_id": gold_row["gold_id"],
                    "gold_kind": gold_kind,
                    "dataset": ds,
                    "prior": prior,
                    "cell_idx": cell_idx,
                    "arm_row": arm_row,
                    "arm_name": arm_name,
                    "estimator": spec["estimator"],
                    "pair": spec["pair"],
                    "coupling": spec["coupling"],
                    "level": spec["level"],
                    "kernel": spec["kernel"],
                    "eta1": eta1,
                    "seed": seed,
                    "seed_idx": j,
                    "replicate": j,
                    **sc,
                    **common_prov,
                }
                append_row(out_path, score)
            cache.add(sfid)
            n_new += 1
            msg = (f"rescore [{ds} prior={prior} {arm_name} rep={j} seed={seed}] "
                   f"eta1={eta1:.4g} n_finite={score_against(theta, arch_gold)['n_finite']}")
            for gk, gv in (("arch", arch_gold), ("hard", hard_gold)):
                s = score_against(theta, gv)
                med = s["mse_gold_median"]
                msg += f" mse_{gk}={'diverged' if med is None else f'{med:.3e}'}"
            if accept_rate is not None:
                msg += f" accept={accept_rate:.3f}"
            print(msg + f" ({elapsed:.1f}s)", flush=True)
    return n_new


# ----------------------------------------------------------------------------
# --run (GPU, gated).
# ----------------------------------------------------------------------------

def run(args):
    for guard in (args.archived_gold, args.hardened_gold):
        if args.out.resolve() == Path(guard).resolve():
            sys.exit(f"refusing to write rescore rows into a gold file {guard}")
    preflight_text, gpu_uuid = gpu_preflight(args.gpu_index)
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if prev is not None and prev != str(args.gpu_index):
        sys.exit(f"CUDA_VISIBLE_DEVICES={prev!r} conflicts with --gpu-index {args.gpu_index}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    import cupy as cp_mod  # after preflight, per section H
    rr.cp = cp_mod

    cells = [(ds, pr) for ds in args.datasets for pr in args.priors]
    passing = [int(t) for t in args.passing_attempts.split(",")]
    if len(passing) != 6 or any(t not in (0, 1) for t in passing):
        sys.exit(f"--passing-attempts must be six 0/1 entries, got {args.passing_attempts!r}")
    arch = load_archived_gold(args.archived_gold)
    for cell in cells:
        if cell not in arch:
            sys.exit(f"archived gold {args.archived_gold} has no row for cell {cell}")
    hard = select_hardened_gold(args.hardened_gold, cells, passing)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = load_cache(args.out)
    script_sha = rr.sha256_file(Path(__file__))
    common_prov = {
        "script_sha256": script_sha,
        "rr_script_sha256": rr.sha256_file(Path(rr.__file__)),
        "git_hash": rr.git_hash(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta = {
        "kind": "rescore_run_meta",
        "schema_version": SCHEMA_VERSION,
        "target": "gold_rescore_p4",
        "precommit": PRECOMMIT,
        "argv": sys.argv,
        "gpu_index": args.gpu_index,
        "gpu_uuid": gpu_uuid,
        "gpu_preflight": preflight_text,
        "seed_base": RESCORE_SEED_BASE,
        "passing_attempts": passing,
        "archived_gold_file": str(args.archived_gold),
        "archived_gold_sha256": rr.sha256_file(args.archived_gold),
        "hardened_gold_file": str(args.hardened_gold),
        "hardened_gold_sha256": rr.sha256_file(args.hardened_gold),
        "hardened_gold_ids": {f"{c[0]}|{c[1]}": hard[c]["gold_id"] for c in cells},
        "hardened_gold_diag": {f"{c[0]}|{c[1]}": {"rhat_max": hard[c].get("rhat_max"),
                                                  "ess_min": hard[c].get("ess_min"),
                                                  "attempt": hard[c].get("attempt")}
                               for c in cells},
        **common_prov,
    }
    append_row(args.out, meta)
    print(f"rescore run: cells={cells} chains={CHAINS} seed_base={RESCORE_SEED_BASE} "
          f"out={args.out} (archived etas frozen; paired scoring vs archived+hardened)",
          flush=True)

    for ds in args.datasets:
        data_x, data_y = rr.load_dataset(args.data_dir, ds)
        cfg = rr.DATASET_CFG[ds]
        data_sha = rr.sha256_file(args.data_dir / f"{ds}.npz")
        for prior in args.priors:
            cell = (ds, prior)
            cell_idx = rr.p4_cell_idx(ds, prior)
            prov = dict(common_prov, data_sha256=data_sha)
            rescore_cell(args.out, ds, prior, cell_idx, arch[cell], hard[cell],
                         data_x, data_y, cfg["n_ens"], CHAINS, cfg["budget"],
                         args.chunk_rows, cache, prov)
    # Final structural check: the file must satisfy the strict loader.
    expected = {c: {"archived": arch[c]["gold_id"], "hardened": hard[c]["gold_id"]}
                for c in cells}
    fits, _ = load_rescore(args.out, SCHEMA_VERSION, expected)
    print(f"rescore complete: {len(fits)} fits x 2 score rows in {args.out} "
          "(strict loader PASS)", flush=True)


# ----------------------------------------------------------------------------
# --selftest (CPU only; cupy never imported).
# ----------------------------------------------------------------------------

def _expect_abort(fn, *a, **kw):
    """True iff fn(*a) raises SystemExit; returns (aborted, message)."""
    try:
        fn(*a, **kw)
        return False, ""
    except SystemExit as exc:
        return True, str(exc)


def selftest():
    rr.cp = rr._CpuCupyShim()
    checks = {}

    # (A) Arm-spec fidelity against the archived generator.
    eta_c, budget = rr.default_eta(0.3), 2400
    arms = build_arm_specs(eta_c, budget)
    a = {name: spec for _, name, spec in arms}
    checks["arm_specs_match_archived_generator"] = (
        [r for r, _, _ in arms] == [0, 1, 2, 3]
        and a["iid_eta1"]["eta1"] == 4 * eta_c and a["iid_eta1"]["coupling"] == "iid"
        and a["zs_eta1"]["eta1"] == 4 * eta_c and a["zs_eta1"]["coupling"] == "zero_sum"
        and a["zs_eta2"]["eta1"] == 2 * eta_c and a["zs_eta2"]["level"] == "eta2"
        and all(a[n]["steps1"] == budget and a[n]["kernel"] == "ula"
                and a[n]["estimator"] == "single" and a[n]["pair"] == "primary"
                for n in ["iid_eta1", "zs_eta1", "zs_eta2"])
        and a["zs_mala"]["estimator"] == "mala" and a["zs_mala"]["kernel"] == "mala"
        and a["zs_mala"]["coupling"] == "zero_sum" and a["zs_mala"]["pair"] == "context"
        and a["zs_mala"]["eta1"] is None and a["zs_mala"]["steps1"] == budget)
    checks["dataset_cfg_frozen_as_archived"] = (
        rr.DATASET_CFG == {"wdbc": {"n_ens": 128, "budget": 2400},
                           "spambase": {"n_ens": 128, "budget": 2400},
                           "adult": {"n_ens": 64, "budget": 1800}}
        and CHAINS == 8
        and abs(rr.default_eta(0.1) - 0.08) < 1e-15
        and abs(rr.default_eta(0.3) - 0.03 / 0.55) < 1e-15)

    # (B) Registered 344M seed map: spot values, distinctness, disjointness
    # from the published P4 345M init / 346M dynamics maps.
    seeds = all_rescore_seeds()
    checks["seed_map_spot_values"] = (
        rescore_seed(0, 0, 0) == 344000000 and rescore_seed(2, 1, 1) == 344211000
        and rescore_seed(5, 3, 2) == 344532000)
    checks["seed_map_72_distinct_in_344M_block"] = (
        len(seeds) == 72 and len(set(seeds)) == 72
        and min(seeds) == 344000000 and max(seeds) == 344532000
        and all(344000000 <= s < 345000000 for s in seeds))
    _, init_all, dyn_all = rr.p4_seed_map_lines()
    checks["seed_map_disjoint_from_p4_345M_init_346M_dyn"] = (
        len(init_all) == 108 and len(dyn_all) == 108
        and not set(seeds) & set(init_all) and not set(seeds) & set(dyn_all))

    # (C) Real gold files: pinned archived sha + hardened selection state.
    arch_path = REPO_ROOT / ARCHIVED_GOLD_DEFAULT
    hard_path = REPO_ROOT / HARDENED_GOLD_DEFAULT
    arch = load_archived_gold(arch_path)
    all_cells = [(ds, pr) for ds in rr.DATASETS for pr in [0.1, 0.3]]
    checks["archived_gold_sha_pinned_and_6_cells"] = (
        set(arch) == set(all_cells) and arch[("wdbc", 0.1)]["mala_eta"] == 3.2)
    five = [c for c in all_cells if c != ("adult", 0.3)]
    hard5 = select_hardened_gold(hard_path, five, PASSING_ATTEMPT_DEFAULT)
    checks["hardened_selection_five_passing_cells_chol_att0"] = (
        set(hard5) == set(five)
        and all(hard5[c]["attempt"] == 0 and hard5[c]["start_mode"] == "chol"
                for c in five))
    aborted, msg = _expect_abort(select_hardened_gold, hard_path, all_cells,
                                 PASSING_ATTEMPT_DEFAULT)
    if aborted:
        state = "PENDING (adult/0.3 attempt-1 chol row not landed; selection aborts)"
        checks["hardened_selection_adult03_fail_closed_or_landed"] = "adult" in msg
    else:
        state = "LANDED (adult/0.3 attempt-1 chol row present; six-cell selection passes)"
        checks["hardened_selection_adult03_fail_closed_or_landed"] = True
    print(f"hardened-gold remediation state: {state}")

    # Synthetic selection aborts: missing cell and duplicate row.
    with tempfile.TemporaryDirectory() as td:
        synth = Path(td) / "synth_p4.jsonl"
        base = {"kind": "gold", "schema_version": HARDENED_SCHEMA_VERSION,
                "start_mode": "chol", "arm_idx": 1, "attempt": 0, "dataset": "wdbc",
                "prior": 0.1, "gold_id": "g1", "dim": 4, "mala_eta": 0.4,
                "gold_mean": [0.0, 0.0, 0.0, 0.0]}
        synth.write_text(json.dumps(base) + "\n")
        ab_missing, _ = _expect_abort(select_hardened_gold, synth,
                                      [("wdbc", 0.1), ("wdbc", 0.3)], PASSING_ATTEMPT_DEFAULT)
        synth.write_text(json.dumps(base) + "\n" + json.dumps(base) + "\n")
        ab_dup, _ = _expect_abort(select_hardened_gold, synth, [("wdbc", 0.1)],
                                  PASSING_ATTEMPT_DEFAULT)
        bad = dict(base, schema_version="wrong")
        synth.write_text(json.dumps(bad) + "\n")
        ab_schema, _ = _expect_abort(select_hardened_gold, synth, [("wdbc", 0.1)],
                                     PASSING_ATTEMPT_DEFAULT)
        checks["hardened_selection_aborts_missing_dup_schema"] = (
            ab_missing and ab_dup and ab_schema)

    # (D) End-to-end synthetic paired rescore on the CPU shim.
    rs = np.random.RandomState(20260718)
    n_pts, d_feat = 40, 3
    xf = rs.standard_normal((n_pts, d_feat)).astype(np.float32)
    logits = 0.7 * xf[:, 0] - 0.4 * xf[:, 1] + 0.2
    y_np = (rs.random_sample(n_pts) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    data_x = rr.cp.asarray(rr.add_intercept(xf), dtype=rr.cp.float32)
    data_y = rr.cp.asarray(y_np, dtype=rr.cp.float32)
    dim, prior = d_feat + 1, 0.3
    n_ens_t, chains_t, budget_t = 12, 6, 240
    gold_a = 0.1 * np.arange(dim, dtype=np.float64)
    gold_h = gold_a + 0.05
    arch_row = {"gold_id": "synthA|arch", "gold_mean": [float(v) for v in gold_a],
                "dim": dim, "mala_eta": 0.4}
    hard_row = {"gold_id": "synthA|hard", "gold_mean": [float(v) for v in gold_h],
                "dim": dim, "mala_eta": 0.4}
    prov = {"script_sha256": "selftest", "git_hash": "selftest", "gpu": "cpu-shim",
            "data_sha256": "selftest", "timestamp_utc": "selftest",
            "rr_script_sha256": "selftest"}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "rescore.jsonl"
        cache = load_cache(out)
        n_new = rescore_cell(out, "wdbc", prior, 0, arch_row, hard_row, data_x, data_y,
                             n_ens_t, chains_t, budget_t, 1024, cache, prov,
                             seed_idxs=(0, 1))
        expected_ids = {("wdbc", prior): {"archived": "synthA|arch",
                                          "hardened": "synthA|hard"}}
        fits, scores = load_rescore(out, SCHEMA_VERSION, expected_ids)
        checks["run_writes_fit_plus_two_scores_per_arm"] = (
            n_new == 8 and len(fits) == 8
            and all(len(v) == 2 for v in scores.values()))

        # Paired-contrast bookkeeping: recompute BOTH scores from the retained
        # (JSON round-tripped) per-ensemble means; bitwise-equal to stored.
        pair_ok, share_ok, recompute_ok = True, True, True
        for sfid, fit in fits.items():
            theta = np.asarray(fit["theta_ens"], dtype=np.float64)
            s_arch = scores[sfid]["synthA|arch"]
            s_hard = scores[sfid]["synthA|hard"]
            for srow, gvec in ((s_arch, gold_a), (s_hard, gold_h)):
                rec = score_against(theta, gvec)
                if any(rec[k] != srow[k] for k in
                       ["mse_gold_median", "mse_gold_mean", "bias_sq_gold",
                        "trace_var", "n_finite"]):
                    recompute_ok = False
            if not (s_arch["trace_var"] == s_hard["trace_var"]
                    and s_arch["n_finite"] == s_hard["n_finite"] == n_ens_t):
                share_ok = False
            if s_arch["mse_gold_median"] == s_hard["mse_gold_median"]:
                pair_ok = False  # distinct golds must yield distinct scores here
        checks["paired_contrast_recompute_bitwise_from_retained_means"] = recompute_ok
        checks["score_pair_shares_estimates_trace_var_n_finite"] = share_ok
        checks["score_pair_distinguishes_the_two_references"] = pair_ok

        mala_fits = [f for f in fits.values() if f["arm_name"] == "zs_mala"]
        ula_fits = [f for f in fits.values() if f["arm_name"] != "zs_mala"]
        checks["mala_arm_uses_archived_eta_and_records_acceptance"] = (
            all(f["eta1"] == 0.4 and f["accept_rate"] is not None
                and 0.0 < f["accept_rate"] <= 1.0 for f in mala_fits)
            and all(f["accept_rate"] is None for f in ula_fits))

        # Resume: a second call is a full cache hit and leaves the file
        # byte-identical.
        before = out.read_bytes()
        cache2 = load_cache(out)
        n_again = rescore_cell(out, "wdbc", prior, 0, arch_row, hard_row, data_x, data_y,
                               n_ens_t, chains_t, budget_t, 1024, cache2, prov,
                               seed_idxs=(0, 1))
        checks["resume_full_cache_hit_file_unchanged"] = (
            n_again == 0 and out.read_bytes() == before)

        # Determinism: same registered seed -> bitwise-identical theta.
        spec_iid = build_arm_specs(rr.default_eta(prior), budget_t)[0][2]
        t1, _, _, _ = run_arm(spec_iid, rescore_seed(0, 0, 0), 0.4, n_ens_t, chains_t,
                              dim, prior, data_x, data_y, 1024)
        t2, _, _, _ = run_arm(spec_iid, rescore_seed(0, 0, 0), 0.4, n_ens_t, chains_t,
                              dim, prior, data_x, data_y, 1024)
        checks["determinism_same_seed_bitwise_theta"] = bool(np.array_equal(t1, t2))

        # (E) Loader aborts on corrupted copies.
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        some_score = next(r for r in lines if r["kind"] == "rescore_score")
        some_fit = next(r for r in lines if r["kind"] == "rescore_fit")

        def write_variant(rows):
            p = Path(td) / "variant.jsonl"
            p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
            return p

        v = write_variant(lines + [some_score])
        checks["loader_aborts_on_duplicate_score_key"], _ = _expect_abort(
            load_rescore, v, SCHEMA_VERSION, expected_ids)
        v = write_variant([r for r in lines
                           if not (r["kind"] == "rescore_score"
                                   and r["source_fit_id"] == some_score["source_fit_id"]
                                   and r["score_gold_id"] == some_score["score_gold_id"])])
        checks["loader_aborts_on_single_score_row"], _ = _expect_abort(
            load_rescore, v, SCHEMA_VERSION, expected_ids)
        third = dict(some_score, score_gold_id="synthA|third")
        v = write_variant(lines + [third])
        checks["loader_aborts_on_three_score_rows"], _ = _expect_abort(
            load_rescore, v, SCHEMA_VERSION, expected_ids)
        bad_sv = [dict(r) for r in lines]
        bad_sv[3]["schema_version"] = "p4-0"
        v = write_variant(bad_sv)
        checks["loader_aborts_on_wrong_schema_version"], _ = _expect_abort(
            load_rescore, v, SCHEMA_VERSION, expected_ids)
        wrong_map = {("wdbc", prior): {"archived": "synthA|arch",
                                       "hardened": "SOMETHING_ELSE"}}
        checks["loader_aborts_on_unexpected_score_gold_id"], _ = _expect_abort(
            load_rescore, out, SCHEMA_VERSION, wrong_map)
        orphan = dict(some_score, source_fit_id="p4rescore|ghost")
        v = write_variant(lines + [orphan, dict(orphan, score_gold_id="synthA|hard")])
        ab_orphan, _ = _expect_abort(load_rescore, v, SCHEMA_VERSION, expected_ids)
        v = write_variant(lines + [some_fit])
        ab_dupfit, _ = _expect_abort(load_rescore, v, SCHEMA_VERSION, expected_ids)
        checks["loader_aborts_on_orphan_score_and_duplicate_fit"] = ab_orphan and ab_dupfit

        # Fail-closed resume: a fit without its two score rows aborts the cache.
        v = write_variant(lines + [dict(some_fit, source_fit_id="p4rescore|partial")])
        checks["resume_aborts_on_partial_triple"], _ = _expect_abort(load_cache, v)

    # (F) Section-H preflight decision function (pure, canned outputs).
    gcsv = "0, GPU-aaa\n1, GPU-bbb\n2, GPU-ccc\n"
    ok_free, _ = _preflight_decide(gcsv, "", 1)
    ok_other, _ = _preflight_decide(gcsv, "GPU-ccc, 123, python\n", 1)
    bad_busy, _ = _preflight_decide(gcsv, "GPU-bbb, 123, wave2d\n", 1)
    bad_idx, _ = _preflight_decide(gcsv, "", 7)
    bad_parse, _ = _preflight_decide(gcsv, "garbage line\n", 1)
    bad_gpu, _ = _preflight_decide("nonsense\n", "", 0)
    checks["preflight_fail_closed_decisions"] = (
        ok_free and ok_other and not bad_busy and not bad_idx and not bad_parse
        and not bad_gpu)

    print("\nselftest checks:")
    for k, val in checks.items():
        print(f"  {'PASS' if val else 'FAIL'}  {k}")
    verdict = all(checks.values())
    print(f"selftest {'PASS' if verdict else 'FAIL'} "
          f"({sum(bool(v) for v in checks.values())}/{len(checks)})")
    return 0 if verdict else 1


# ----------------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true",
                        help="CPU-only unit checks (numpy shim; never imports cupy)")
    parser.add_argument("--run", action="store_true",
                        help="accepted V3-5 rescore run (GPU; section-H preflight)")
    parser.add_argument("--gpu-index", type=int, default=None,
                        help="physical GPU index for the section-H preflight (required with --run)")
    parser.add_argument("--datasets", nargs="+", choices=rr.DATASETS, default=rr.DATASETS)
    parser.add_argument("--priors", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--archived-gold", type=Path, default=ARCHIVED_GOLD_DEFAULT)
    parser.add_argument("--hardened-gold", type=Path, default=HARDENED_GOLD_DEFAULT)
    parser.add_argument("--passing-attempts", default=",".join(str(t) for t in PASSING_ATTEMPT_DEFAULT),
                        help="per-cell passing attempt, cell_idx order 0..5 "
                             "(default per the 2026-07-18 remediation ruling)")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    if args.selftest == args.run:
        sys.exit("exactly one of --selftest / --run is required")
    if args.selftest:
        sys.exit(selftest())
    if args.gpu_index is None:
        sys.exit("--run requires --gpu-index (section-H preflight)")
    run(args)


if __name__ == "__main__":
    main()
