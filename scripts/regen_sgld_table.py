#!/usr/bin/env python3
"""G4 generator: regenerate paper/tables/minibatch_sgld_real.tex from raw JSONLs.

Registered contract (camera-ready regeneration campaign):
  - campaigns/camera_ready_regen_invocation_plan.md — SECTION 2 common
    requirements + "### G4" spec (Revision 5).
  - campaigns/camera_ready_regen_precommit.md — DR-4 (hardened generator
    contract), DR-7 (completed match rule), VR-3 (per-mode schema pins),
    VR-6/VR-9 (archived common_batch_stream absence semantics), AM-7
    (harness version-skew disclosure), VR-2 (seed-sharing disclosure).

Deterministic, CPU-only. Fail-closed: ANY check failure => no output file is
written and the exit code is nonzero.

Modes:
  --mode archived
      Inputs are EXACTLY the registered archived June raws:
        campaigns/attempt03/{primary_rep30..35,stress_rep36..41,edge_rep42..44}.jsonl
        campaigns/minibatch_followup_allcells/allcells_rep{60..65}.jsonl
        campaigns/minibatch_followup/common_stream_rep{70,71,72}.jsonl
      Gate (DR-4b): the emitted table must be BYTE-IDENTICAL to the committed
      paper/tables/minibatch_sgld_real.tex.  On PASS a scratch copy is written
      under --scratch-dir; paper/tables/ is NEVER written.
      NOTE: campaigns/minibatch_followup/allcells_rep6*.jsonl is a
      divergent same-seed duplicate registered as EXCLUDED (plan ### G4;
      GPU-nondeterminism precedent, AM-8).  This script never reads it: the
      archived input paths are hard-pinned above, never globbed.
  --mode fresh
      First re-runs the entire archived phase in memory and requires the
      byte-identity gate to PASS (SECTION 2: the generator is not trusted on
      fresh raws otherwise).  Then validates the fresh raws under the FRESH
      pins (VR-3: 46 keys in ALL rows of ALL three families; exact per-row
      seed/design_seed recomputation for replicates 40229-40252 per DR-4d) and
      emits, under --out-root (default the registered DR-4e root
      outputs/sweeps/camera_ready_regen/sgld/tables/):
        minibatch_sgld_real_fresh.tex
        comparison_sgld.json   (AM-13 + DR-7 match rule; AM-7/VR-2 notes)

Note for this repository: the fresh-mode inputs and the protocol documents
named above are not included.  Archived mode needs only the files in this
repository.  Host names are redacted in the released rows, so host checks
compare against the redaction token.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_TEX = REPO_ROOT / "paper" / "tables" / "minibatch_sgld_real.tex"

DATASETS = ("wdbc", "spambase", "adult")
DISPLAY = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}
DIM = {"wdbc": 31, "spambase": 58, "adult": 105}
SEED_BASE = {"wdbc": 710000, "spambase": 720000, "adult": 730000}
STEPS = {"wdbc": 3600, "spambase": 3600, "adult": 2800}
BURNIN = {"wdbc": 1800, "spambase": 1800, "adult": 1400}
REPS = {"wdbc": 256, "spambase": 192, "adult": 64}
PRIMARY_BATCH = {"wdbc": 8, "spambase": 64, "adult": 512}
SMALL_BATCH = {"wdbc": 4, "spambase": 32, "adult": 256}
PRIORS = (0.1, 0.3)
CHAINS = 8
REF_REPS = 16
REF_CHAINS = 16

METHODS5 = ("iid_iid", "iid_zero", "balanced_iid", "balanced_zero", "iid_antithetic")
METHODS3 = ("iid_iid", "iid_zero", "iid_antithetic")
METHOD_PARTS = {
    "iid_iid": ("iid", "iid"),
    "iid_zero": ("iid", "zero_sum"),
    "balanced_iid": ("balanced", "iid"),
    "balanced_zero": ("balanced", "zero_sum"),
    "iid_antithetic": ("iid", "antithetic"),
}

# VR-3 schema pins, enumerated from the archived raws (42-key attempt03-style
# set verified identical across all 15 attempt03 files; 45 = 46 minus
# common_batch_stream; 46 = 42 plus the four block-specific fields, identical
# between archived allcells/common rows and the HEAD harness emission).
KEYS_42 = frozenset((
    "batch_policy", "batch_regime", "batch_size", "bias_sq", "burnin",
    "chains", "coupling", "dataset", "design_seed", "dim", "elapsed_sec",
    "elapsed_sec_per_ensemble", "eta", "full_gradient_eta", "git_hash", "gpu",
    "lin_chain_var", "lin_ens_var", "marginal_second", "method", "mse",
    "mse_to_single", "n_ensembles", "prior", "ref_burnin", "ref_chains",
    "ref_reps", "ref_steps", "replicate", "reps", "seed", "single_bias_sq",
    "single_mse", "single_test_logloss", "single_trace_var", "steps",
    "target", "terminal_spread", "test_logloss", "test_logloss_var",
    "trace_var", "windows",
))
BLOCK_FIELDS = frozenset(("iid_batch_replacement", "common_batch_stream", "init_mode", "init_scale"))
KEYS_46 = KEYS_42 | BLOCK_FIELDS
KEYS_45 = KEYS_46 - frozenset(("common_batch_stream",))

INT_FIELDS = frozenset((
    "dim", "chains", "steps", "burnin", "ref_steps", "ref_burnin", "ref_reps",
    "ref_chains", "reps", "replicate", "batch_size", "seed", "design_seed",
    "n_ensembles", "windows",
))
STR_FIELDS = frozenset((
    "dataset", "batch_regime", "target", "method", "batch_policy", "coupling",
    "gpu", "git_hash", "init_mode",
))
FLOAT_FIELDS = frozenset((
    "prior", "eta", "full_gradient_eta", "elapsed_sec",
    "elapsed_sec_per_ensemble", "mse", "bias_sq", "trace_var", "single_mse",
    "single_bias_sq", "single_trace_var", "mse_to_single", "lin_ens_var",
    "lin_chain_var", "test_logloss", "single_test_logloss",
    "test_logloss_var", "marginal_second", "terminal_spread", "init_scale",
))
BOOL_FIELDS = frozenset(("iid_batch_replacement", "common_batch_stream"))

# family -> (batch regime, method set, rows per file)
FAMILY_CFG = {
    "primary": ("primary", METHODS5, 30),
    "stress": ("primary", METHODS5, 30),
    "edge": ("small", METHODS5, 30),
    "allcells": ("primary", METHODS3, 18),
    "common": ("primary", METHODS3, 18),
}
ATTEMPT03_STYLE = ("primary", "stress", "edge")
# (family, column header) in frozen column order.
COLUMNS = (
    ("primary", "primary"),
    ("stress", "stress"),
    ("edge", "small-batch"),
    ("allcells", "random start/WR"),
    ("common", "common stream"),
)
FAMILY_ORDER = tuple(f for f, _ in COLUMNS)

ARCHIVED_REPLICATES = {
    "primary": tuple(range(30, 36)),
    "stress": tuple(range(36, 42)),
    "edge": tuple(range(42, 45)),
    "allcells": tuple(range(60, 66)),
    "common": tuple(range(70, 73)),
}
FRESH_REPLICATES = {
    "primary": tuple(range(40229, 40235)),
    "stress": tuple(range(40235, 40241)),
    "edge": tuple(range(40241, 40244)),
    "allcells": tuple(range(40244, 40250)),
    "common": tuple(range(40250, 40253)),
}

# Block-specific field expectations (DR-4c), pinned from the archived raws
# (archived) and from the HEAD harness defaults + the three launch wrappers
# run_attempt03_rep.sh / run_minibatch_followup_{allcells,common_stream}_rep.sh
# (fresh): attempt03-style fresh rows carry the harness defaults
# (iid_batch_replacement False, common_batch_stream False, init_mode "zero",
# init_scale 0.0); followup-style rows carry --iid-batch-replacement
# --init-mode iid_normal --init-scale 0.25, common-stream adds
# --common-batch-stream.
FRESH_BLOCK_EXPECT = {
    "primary": {"iid_batch_replacement": False, "common_batch_stream": False, "init_mode": "zero", "init_scale": 0.0},
    "stress": {"iid_batch_replacement": False, "common_batch_stream": False, "init_mode": "zero", "init_scale": 0.0},
    "edge": {"iid_batch_replacement": False, "common_batch_stream": False, "init_mode": "zero", "init_scale": 0.0},
    "allcells": {"iid_batch_replacement": True, "common_batch_stream": False, "init_mode": "iid_normal", "init_scale": 0.25},
    "common": {"iid_batch_replacement": True, "common_batch_stream": True, "init_mode": "iid_normal", "init_scale": 0.25},
}
ARCHIVED_FOLLOWUP_EXPECT = {
    "allcells": {"iid_batch_replacement": True, "init_mode": "iid_normal", "init_scale": 0.25},
    "common": {"iid_batch_replacement": True, "common_batch_stream": True, "init_mode": "iid_normal", "init_scale": 0.25},
}

HEADER_LINE = r"Dataset & prior & batch & primary & stress & small-batch & random start/WR & common stream \\"
RE_4DP = re.compile(r"^\d+\.\d{4}$")
RE_SCI = re.compile(r"^\$(\d+\.\d{2})\{\\times\}10\^\{-4\}\$$")
MAX_ERRORS_PER_FILE = 3

AM7_NOTE = (
    "AM-7 version-skew disclosure (as extended by Amendment v7): the archived attempt03 "
    "primary/stress/small-batch blocks ran the 328f9b0-era RNG layout (a single "
    "cp.random.seed(seed+101*(m+1)) stream interleaving batch and noise draws) and emit 42-key rows; the "
    "surviving split-stream harness campaigns/real_logistic_sgld_gpu.py (batch seed+101*m+13, noise "
    "seed+101*m+29) always emits iid_batch_replacement/common_batch_stream/init_mode/init_scale, so ALL "
    "fresh SGLD rows carry the 46-key schema (VR-3). TWO fresh harness states apply (Amendment v7): the "
    "fresh allcells/common-stream rows landed under sha256[:16] 11221e3bb3c55e31 (byte-identical to "
    "93dec29); the fresh attempt03-family rows run the v7-fixed harness sha256[:16] a23bc12d1d5af259 "
    "(two-token latent-defect fix, lines 106/115 rng.random -> rng.random_sample, identical "
    "distribution/stream semantics; the pre-fix no-replacement/balanced paths could not execute on CuPy "
    "at all, so no numerical result ever depended on the defective lines). Statistically the same design; "
    "disclosed in the REPRODUCIBILITY note. (The archived allcells/common-stream blocks ran 07a996b, "
    "identical to 11221e3bb3c55e31 at K=8.)"
)
VR2_NOTE = (
    "VR-2 seed-sharing disclosure: no blanket seed-distinctness claim is registered; the authoritative "
    "property is exact per-row seed recomputation (DR-4d), enforced fail-closed by this generator. Two "
    "disclosed sharing categories: (1) INTENTIONAL same-design sharing by harness construction — "
    "common-stream rows share the batch seed seed+90917 across all three methods (e.g. wdbc rep 40250 "
    "prior 0 consumes 403,300,917 for all three) and the reference stream seed+77717 is shared across "
    "methods within a cell; (2) ACCIDENTAL cross-target numeric coincidences — dataset seed bases sit "
    "10000 apart (one replicate step), so adjacent-dataset/adjacent-replicate derived seeds coincide "
    "across targets (e.g. method-1 batch seed 403,010,114 is shared by spambase rep 40229 and wdbc rep "
    "40230; noise seed 403,010,130 likewise; DR-3, AM-10 extended). Both categories reproduce the June "
    "originals' structure; the four campaign envelopes remain disjoint."
)
MATCH_RULE_TEXT = (
    "Registered AM-13 + DR-7 rule (result-review corrected 2026-07-19): every SGLD scalar cell HAS "
    "underlying per-replicate values, so the fresh min-max clause IS the registered verdict for all 30 "
    "cells — the 15% relative quantity |fresh_median - frozen|/|frozen| (DR-7b) is reported ONLY as a "
    "separate descriptive sensitivity quantity (rel_err_vs_frozen), never as a pass/fail fallback. "
    "Full-precision frozen comparanda recomputed from the archived raws via this byte-identity-gated "
    "generator (DR-7a); every cell reported either way; pass fractions descriptive only (DR-8). "
    "Layer framing per AM-0: the archived-mode byte gate is the RESTORATION layer; this record is the "
    "CONFIRMATION layer. AM-8 context: exact device-level replay was never claimed — the June record "
    "itself contains same-seed WDBC allcells duplicates disagreeing (approx. 0.0059-0.0081 across the "
    "two campaigns), which directly contextualizes the wdbc/0.3 allcells miss; that cell comes from the "
    "unaffected replacement branch and is NOT attributable to the v7 two-token fix."
)


class ValidationError(Exception):
    pass


def sha256_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reject_nonfinite(token):
    raise ValueError(f"nonfinite JSON constant {token!r} rejected (fail-closed)")


def _reject_dup_keys(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate JSON keys rejected (fail-closed)")
    return dict(pairs)


def expected_eta(prior):
    # Exactly the harness formula (campaigns/real_logistic_sgld_gpu.py::make_designs).
    full_eta = min(0.03 / (prior + 0.25), 0.08)
    return 0.25 * full_eta, full_eta


def allowed_keysets(mode, family):
    if mode == "fresh":
        # VR-3: fresh pin = 46 keys for ALL rows in ALL three file families.
        return (KEYS_46,)
    if family in ATTEMPT03_STYLE:
        return (KEYS_42,)
    if family == "allcells":
        # VR-6(b)/VR-9: archived allcells rows are 45-key (common_batch_stream
        # absent, e.g. allcells_rep60.jsonl rows 1-6) OR 46-key (field present,
        # value False — row 7 shape, an ordinary valid value).
        return (KEYS_45, KEYS_46)
    return (KEYS_46,)  # common-stream


def keyset_error(keys, pins):
    parts = []
    for pin in pins:
        missing = sorted(pin - keys)
        extra = sorted(keys - pin)
        parts.append(f"vs {len(pin)}-key pin: missing={missing} extra={extra}")
    return f"schema pin violation: row has {len(keys)} keys; " + "; ".join(parts)


def validate_row(mode, family, expect_rep, row):
    """Return a list of error strings (empty if the row is valid)."""
    regime, methods, _ = FAMILY_CFG[family]
    errs = []
    keys = frozenset(row)
    pins = allowed_keysets(mode, family)
    if keys not in pins:
        errs.append(keyset_error(keys, pins))

    for k, v in row.items():
        if k in INT_FIELDS and type(v) is not int:
            errs.append(f"field {k}: expected int, got {type(v).__name__}")
        elif k in STR_FIELDS and type(v) is not str:
            errs.append(f"field {k}: expected str, got {type(v).__name__}")
        elif k in FLOAT_FIELDS:
            if type(v) is not float:
                errs.append(f"field {k}: expected float, got {type(v).__name__}")
            elif not math.isfinite(v):
                errs.append(f"field {k}: nonfinite value {v!r} rejected")
        elif k in BOOL_FIELDS and type(v) is not bool:
            errs.append(f"field {k}: expected bool, got {type(v).__name__}")

    ds = row.get("dataset")
    prior = row.get("prior")
    if ds not in DATASETS:
        errs.append(f"dataset {ds!r} not in {DATASETS}")
        return errs
    if prior not in PRIORS:
        errs.append(f"prior {prior!r} not in {PRIORS}")
        return errs
    prior_idx = PRIORS.index(prior)
    eta, full_eta = expected_eta(prior)

    def chk(name, expected):
        if name in row and row[name] != expected:
            errs.append(f"{name}={row[name]!r} != expected {expected!r}")

    chk("target", "real_logistic_sgld")
    chk("dim", DIM[ds])
    chk("chains", CHAINS)  # K=8
    chk("eta", eta)
    chk("full_gradient_eta", full_eta)
    chk("steps", STEPS[ds])
    chk("burnin", BURNIN[ds])
    chk("windows", STEPS[ds] - BURNIN[ds])
    chk("ref_steps", 2 * STEPS[ds])
    chk("ref_burnin", 2 * BURNIN[ds])
    chk("ref_reps", REF_REPS)
    chk("ref_chains", REF_CHAINS)
    chk("reps", REPS[ds])
    chk("n_ensembles", REPS[ds])
    chk("batch_regime", regime)
    chk("batch_size", (PRIMARY_BATCH if regime == "primary" else SMALL_BATCH)[ds])
    # Exact per-row seed recomputation (DR-4d; replaces interval checks).
    chk("replicate", expect_rep)
    chk("seed", SEED_BASE[ds] + 10000 * expect_rep + prior_idx)
    chk("design_seed", SEED_BASE[ds] + prior_idx)

    method = row.get("method")
    if method not in methods:
        errs.append(f"method {method!r} not in registered {family} set {methods}")
    else:
        chk("batch_policy", METHOD_PARTS[method][0])
        chk("coupling", METHOD_PARTS[method][1])

    # Block-specific field rules (DR-4c) under the per-mode pins.
    if mode == "fresh":
        for name, expected in FRESH_BLOCK_EXPECT[family].items():
            chk(name, expected)
    elif family in ATTEMPT03_STYLE:
        # Archived attempt03-style rows: the four block fields are ABSENT
        # (42-key pin above); absent common_batch_stream => False (VR-6).
        for name in BLOCK_FIELDS:
            if name in row:
                errs.append(f"{name} present in archived {family} row (must be absent per 42-key pin)")
    else:
        for name, expected in ARCHIVED_FOLLOWUP_EXPECT[family].items():
            chk(name, expected)
        if family == "allcells" and "common_batch_stream" in row:
            chk("common_batch_stream", False)

    tv = row.get("trace_var")
    if type(tv) is float and math.isfinite(tv) and tv <= 0.0:
        errs.append(f"trace_var={tv!r} not positive")
    for name in ("gpu", "git_hash"):
        if type(row.get(name)) is str and not row[name]:
            errs.append(f"{name} is empty")
    return errs


def family_paths(mode, family, fresh_root):
    if mode == "archived":
        base = {
            "primary": REPO_ROOT / "campaigns" / "attempt03",
            "stress": REPO_ROOT / "campaigns" / "attempt03",
            "edge": REPO_ROOT / "campaigns" / "attempt03",
            "allcells": REPO_ROOT / "campaigns" / "minibatch_followup_allcells",
            "common": REPO_ROOT / "campaigns" / "minibatch_followup",
        }[family]
        reps = ARCHIVED_REPLICATES[family]
    else:
        base = {
            "primary": fresh_root / "attempt03",
            "stress": fresh_root / "attempt03",
            "edge": fresh_root / "attempt03",
            "allcells": fresh_root / "allcells",
            "common": fresh_root / "common",
        }[family]
        reps = FRESH_REPLICATES[family]
    stem = {"primary": "primary_rep", "stress": "stress_rep", "edge": "edge_rep",
            "allcells": "allcells_rep", "common": "common_stream_rep"}[family]
    return [(base / f"{stem}{r}.jsonl", r) for r in reps]


def load_inputs(mode, fresh_root):
    """Load + validate all five families. Returns (families dict, input paths)."""
    errors = []
    families = {}
    all_paths = []
    for family in FAMILY_ORDER:
        regime, methods, nrows = FAMILY_CFG[family]
        grid = {(ds, prior, m) for ds in DATASETS for prior in PRIORS for m in methods}
        per_rep = {}
        for path, rep in family_paths(mode, family, fresh_root):
            all_paths.append(path)
            tag = str(path)
            if not path.is_file():
                errors.append(f"{tag}: missing input file")
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            if len(lines) != nrows:
                errors.append(f"{tag}: {len(lines)} rows != registered {nrows}")
                continue
            ferrs = []
            seen = {}
            rows = []
            for i, line in enumerate(lines, 1):
                try:
                    row = json.loads(line, parse_constant=_reject_nonfinite,
                                     object_pairs_hook=_reject_dup_keys)
                except ValueError as exc:
                    ferrs.append(f"{tag}:{i}: {exc}")
                    continue
                if not isinstance(row, dict):
                    ferrs.append(f"{tag}:{i}: row is not a JSON object")
                    continue
                for e in validate_row(mode, family, rep, row):
                    ferrs.append(f"{tag}:{i}: {e}")
                cell = (row.get("dataset"), row.get("prior"), row.get("method"))
                if cell in seen:
                    ferrs.append(f"{tag}:{i}: duplicate row for {cell} (first at line {seen[cell]})")
                seen[cell] = i
                rows.append(row)
            if not ferrs and set(seen) != grid:
                missing = sorted(str(c) for c in grid - set(seen))
                extra = sorted(str(c) for c in set(seen) - grid)
                ferrs.append(f"{tag}: cell grid mismatch: missing={missing} unexpected={extra}")
            if ferrs:
                errors.extend(ferrs[:MAX_ERRORS_PER_FILE])
                if len(ferrs) > MAX_ERRORS_PER_FILE:
                    errors.append(f"{tag}: ... {len(ferrs) - MAX_ERRORS_PER_FILE} more error(s) suppressed")
                continue
            per_rep[rep] = rows
        families[family] = per_rep
    if errors:
        raise ValidationError(f"{mode.upper()}-mode validation failed "
                              f"({len(errors)} error line(s)):\n" + "\n".join(errors))
    return families, all_paths


def compute_cells(families):
    """cells[(ds, prior, family)] = {per_replicate, median, min, max}."""
    cells = {}
    for family in FAMILY_ORDER:
        for ds in DATASETS:
            for prior in PRIORS:
                per_rep = {}
                for rep in sorted(families[family]):
                    tv = {r["method"]: r["trace_var"] for r in families[family][rep]
                          if r["dataset"] == ds and r["prior"] == prior}
                    ratio = tv["iid_zero"] / tv["iid_iid"]
                    if not (math.isfinite(ratio) and ratio > 0.0):
                        raise ValidationError(
                            f"nonfinite/nonpositive ratio {ratio!r} at ({ds}, {prior}, {family}, rep {rep})")
                    per_rep[rep] = ratio
                vals = [per_rep[r] for r in sorted(per_rep)]
                cells[(ds, prior, family)] = {
                    "per_replicate": per_rep,
                    "median": float(np.median(vals)),
                    "min": min(vals),
                    "max": max(vals),
                }
    return cells


def fmt_value(v):
    """Registered rounding rule: >= 1e-3 -> 4 dp; < 1e-3 -> $X.XX{\\times}10^{-4}$."""
    if not (type(v) is float and math.isfinite(v) and v > 0.0):
        raise ValidationError(f"unformattable cell value {v!r}")
    if v >= 1e-3:
        return f"{v:.4f}"
    return f"${v * 1e4:.2f}{{\\times}}10^{{-4}}$"


def parse_cell(text):
    m = RE_4DP.match(text)
    if m:
        return float(text), "4dp"
    m = RE_SCI.match(text)
    if m:
        return float(m.group(1)) * 1e-4, "sci"
    raise ValidationError(f"cell {text!r} matches neither registered format")


def build_table(cells):
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule", HEADER_LINE, r"\midrule"]
    for di, ds in enumerate(DATASETS):
        if di:
            lines.append(r"\addlinespace")
        for prior in PRIORS:
            vals = [fmt_value(cells[(ds, prior, family)]["median"]) for family in FAMILY_ORDER]
            lines.append(f"{DISPLAY[ds]} & {prior} & {PRIMARY_BATCH[ds]} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def parse_table(text):
    """Structural re-parse of an emitted/frozen table. Returns
    {(ds, prior, family): cell string}. Fail-closed on any deviation."""
    if not text.endswith("\n"):
        raise ValidationError("table does not end with newline")
    lines = text.splitlines()
    static = {0: r"\begin{tabular}{llrrrrrr}", 1: r"\toprule", 2: HEADER_LINE, 3: r"\midrule",
              6: r"\addlinespace", 9: r"\addlinespace", 12: r"\bottomrule", 13: r"\end{tabular}"}
    if len(lines) != 14:
        raise ValidationError(f"table has {len(lines)} lines, expected 14")
    for i, want in static.items():
        if lines[i] != want:
            raise ValidationError(f"table line {i + 1}: {lines[i]!r} != {want!r}")
    row_lines = {("wdbc", 0.1): 4, ("wdbc", 0.3): 5, ("spambase", 0.1): 7,
                 ("spambase", 0.3): 8, ("adult", 0.1): 10, ("adult", 0.3): 11}
    out = {}
    for (ds, prior), li in row_lines.items():
        line = lines[li]
        if not line.endswith(r" \\"):
            raise ValidationError(f"table line {li + 1} does not end with ' \\\\'")
        parts = line[:-3].split(" & ")
        if len(parts) != 8:
            raise ValidationError(f"table line {li + 1}: {len(parts)} columns, expected 8")
        if parts[0] != DISPLAY[ds] or parts[1] != str(prior) or parts[2] != str(PRIMARY_BATCH[ds]):
            raise ValidationError(f"table line {li + 1}: label columns {parts[:3]} unexpected")
        for family, cell_text in zip(FAMILY_ORDER, parts[3:]):
            out[(ds, prior, family)] = cell_text
    return out


def self_checks(cells, table_text, label):
    """Structural self-checks: 6 data rows x 5 ratio columns + batch column;
    re-parse; rounding-rule round-trip."""
    parsed = parse_table(table_text)
    if len(parsed) != 30:
        raise ValidationError(f"{label}: parsed {len(parsed)} ratio cells, expected 30")
    for key, cell_text in parsed.items():
        computed = cells[key]["median"]
        if cell_text != fmt_value(computed):
            raise ValidationError(f"{label}: cell {key}: printed {cell_text!r} != fmt(computed median)")
        value, kind = parse_cell(cell_text)
        if fmt_value(value) != cell_text:
            raise ValidationError(f"{label}: cell {key}: rounding rule not idempotent for {cell_text!r}")
        tol = 5.000001e-5 if kind == "4dp" else 5.000001e-7
        if abs(value - computed) > tol:
            raise ValidationError(f"{label}: cell {key}: |parsed - computed| = "
                                  f"{abs(value - computed):.3e} > {tol:.1e}")
    print(f"SELF-CHECK [{label}]: 6 data rows x 5 ratio columns + batch column (8/64/512) — OK")
    print(f"SELF-CHECK [{label}]: re-parse of emitted .tex matches computed medians (30/30) — OK")
    print(f"SELF-CHECK [{label}]: rounding-rule round-trip (30/30) — OK")


def run_archived_phase():
    """Validate archived raws, recompute the table, enforce the DR-4b
    byte-identity gate, run self-checks. Returns (emitted, cells, paths)."""
    families, paths = load_inputs("archived", None)
    cells = compute_cells(families)
    emitted = build_table(cells)
    if not FROZEN_TEX.is_file():
        raise ValidationError(f"frozen table missing: {FROZEN_TEX}")
    frozen = FROZEN_TEX.read_bytes()
    emitted_bytes = emitted.encode("utf-8")
    if emitted_bytes != frozen:
        n = min(len(emitted_bytes), len(frozen))
        diff_at = next((i for i in range(n) if emitted_bytes[i] != frozen[i]), n)
        raise ValidationError(
            "ARCHIVED GATE FAIL: emitted table is NOT byte-identical to the frozen "
            f"{FROZEN_TEX} (lengths {len(emitted_bytes)} vs {len(frozen)}; first difference at "
            f"byte {diff_at}: emitted ...{emitted_bytes[max(0, diff_at - 20):diff_at + 20]!r} vs "
            f"frozen ...{frozen[max(0, diff_at - 20):diff_at + 20]!r})")
    self_checks(cells, emitted, "archived")
    print("ARCHIVED GATE PASS (byte-identical)")
    print(f"  frozen : sha256 {hashlib.sha256(frozen).hexdigest()}  {FROZEN_TEX}")
    print(f"  emitted: sha256 {hashlib.sha256(emitted_bytes).hexdigest()}  ({len(emitted_bytes)} bytes)")
    return emitted, cells, paths


def run_fresh_phase(args):
    # SECTION 2: the archived byte-identity gate must PASS before the
    # generator is trusted on fresh raws.
    frozen_text, frozen_cells, archived_paths = run_archived_phase()
    frozen_printed = parse_table(frozen_text)

    fresh_root = args.fresh_root
    out_root = args.out_root
    out_root_resolved = out_root.resolve()
    paper_dir = (REPO_ROOT / "paper").resolve()
    if out_root_resolved == paper_dir or paper_dir in out_root_resolved.parents:
        raise ValidationError(f"--out-root {out_root} is inside paper/ — forbidden (DR-4e)")
    out_tex = out_root / "minibatch_sgld_real_fresh.tex"
    out_json = out_root / "comparison_sgld.json"
    if not args.force:
        for p in (out_tex, out_json):
            if p.exists():
                raise ValidationError(f"output {p} already exists (fail-closed; rerun with --force to overwrite)")

    families, fresh_paths = load_inputs("fresh", fresh_root)
    cells = compute_cells(families)
    fresh_text = build_table(cells)
    self_checks(cells, fresh_text, "fresh")

    records = []
    n_match = n_minmax = n_rel = 0
    for ds in DATASETS:
        for prior in PRIORS:
            for family, column in COLUMNS:
                key = (ds, prior, family)
                frozen_full = frozen_cells[key]["median"]
                c = cells[key]
                in_minmax = c["min"] <= frozen_full <= c["max"]
                rel = abs(c["median"] - frozen_full) / abs(frozen_full)
                clause = "fresh_minmax"
                match = in_minmax
                n_minmax += in_minmax
                n_rel += (not in_minmax and rel <= 0.15)
                n_match += match
                records.append({
                    "dataset": ds,
                    "prior": prior,
                    "block": family,
                    "column": column,
                    "frozen_printed": frozen_printed[key],
                    "frozen_full_precision": frozen_full,
                    "fresh_printed": fmt_value(c["median"]),
                    "fresh_median": c["median"],
                    "fresh_min": c["min"],
                    "fresh_max": c["max"],
                    "fresh_per_replicate": {str(r): c["per_replicate"][r] for r in sorted(c["per_replicate"])},
                    "in_fresh_minmax": bool(in_minmax),
                    "rel_err_vs_frozen": rel,
                    "clause_applied": clause,
                    "match": bool(match),
                })
    record = {
        "schema": "regen-comparison-sgld-1",
        "table": "paper/tables/minibatch_sgld_real.tex",
        "campaign": "sgld",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "match_rule": MATCH_RULE_TEXT,
        "cells": records,
        "summary": {
            "n_cells": len(records),
            "n_match": n_match,
            "n_fail": len(records) - n_match,
            "n_pass_registered_minmax": n_minmax,
            "n_miss_within_15pct_sensitivity": n_rel,
            "pass_fractions_descriptive_only": True,
        },
        "notes": [AM7_NOTE, VR2_NOTE],
        "provenance": {
            "generator": "scripts/regen_sgld_table.py",
            "generator_sha256": sha256_path(__file__),
            "argv": sys.argv,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "frozen_table_sha256": sha256_path(FROZEN_TEX),
            "archived_inputs_sha256": {str(p.relative_to(REPO_ROOT)): sha256_path(p) for p in archived_paths},
            "fresh_inputs_sha256": {str(p): sha256_path(p) for p in fresh_paths},
        },
    }

    out_root.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(fresh_text, encoding="utf-8")
    def _relpaths(o):
        if isinstance(o, dict):
            return {_relpaths(k): _relpaths(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_relpaths(v) for v in o]
        if isinstance(o, str) and o.startswith(str(REPO_ROOT) + "/"):
            return o[len(str(REPO_ROOT)) + 1:]
        return o
    record = _relpaths(record)
    out_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("FRESH MODE COMPLETE")
    print(f"  fresh table      : {out_tex}  (sha256 {sha256_path(out_tex)})")
    print(f"  comparison record: {out_json}")
    print(f"  cells matched    : {n_match}/{len(records)} under the REGISTERED min-max rule "
          f"(descriptive only per DR-8); sensitivity note: {n_rel} of the {len(records) - n_match} "
          f"misses fall within 15% relative of the frozen value")
    for r in records:
        if not r["match"]:
            print(f"  MISMATCH {r['dataset']}/{r['prior']}/{r['block']}: frozen {r['frozen_full_precision']!r} "
                  f"vs fresh median {r['fresh_median']!r} [{r['fresh_min']!r}, {r['fresh_max']!r}], "
                  f"rel {r['rel_err_vs_frozen']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=("archived", "fresh"))
    ap.add_argument("--fresh-root", type=Path,
                    default=REPO_ROOT / "outputs" / "sweeps" / "camera_ready_regen" / "sgld",
                    help="root of the fresh raws (registered AM-3 default; "
                         "overridable for the registered fail-closed demonstration)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="fresh-mode output dir (default <fresh-root>/tables per DR-4e)")
    ap.add_argument("--scratch-dir", type=Path, default=Path("/tmp/regen_g4"),
                    help="archived-mode scratch dir for the emitted-copy byte artifact")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting existing fresh-mode outputs")
    args = ap.parse_args()
    if args.out_root is None:
        args.out_root = args.fresh_root / "tables"
    try:
        if args.mode == "archived":
            emitted, _cells, _paths = run_archived_phase()
            args.scratch_dir.mkdir(parents=True, exist_ok=True)
            copy = args.scratch_dir / "minibatch_sgld_real_archived.tex"
            copy.write_text(emitted, encoding="utf-8")
            print(f"  scratch copy     : {copy}")
        else:
            run_fresh_phase(args)
    except ValidationError as exc:
        print(f"FAIL-CLOSED [{args.mode}]: {exc}", file=sys.stderr)
        print("No output written; exiting nonzero.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
