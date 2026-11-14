#!/usr/bin/env python3
"""G2 deterministic generator — regenerates paper/tables/real_odd_k_summary.tex (ODDK).

Registered under:
  campaigns/camera_ready_regen_invocation_plan.md
      SECTION 2 common requirements (Revision 5) + spec "### G2".
  campaigns/camera_ready_regen_precommit.md
      AM-1b (fresh replicate map = archived 9-23 shifted +40032, envelope
      [401,020,000 .. 401,257,718]), AM-9 (probe design_seed), AM-13 + DR-7
      (match rule), AM-14 (this path), DR-4(a-e) (schema pins, byte gate,
      validator battery, exact seed recomputation, fresh output root),
      VR-2 (seed-sharing disclosure categories), VR-3 (ODDK fresh key-set
      pin == archived pin).

Modes (both deterministic, CPU-only, fail-closed: ANY check failure => no
output file written, nonzero exit):

  --mode archived
      Inputs campaigns/attempt02/ (6 K=7 primary reps 9-14, 6 K=7
      stress reps 15-20, 3 K=5 edge reps 21-23; 18 rows/file, 270 total).
      Gate: emitted .tex BYTE-IDENTICAL to the committed
      paper/tables/real_odd_k_summary.tex.  The gate artifact is written
      under --scratch-dir only — NEVER paper/tables/.

  --mode fresh
      Re-runs the archived gate first (dual-mode trust rule, SECTION 2),
      then validates outputs/sweeps/camera_ready_regen/oddk/ (replicates
      40041-40055) with exact per-row seed/design_seed recomputation
      (DR-4d) and emits
        outputs/sweeps/camera_ready_regen/oddk/tables/real_odd_k_summary_fresh.tex
        outputs/sweeps/camera_ready_regen/oddk/tables/comparison_oddk.json
      under the registered AM-13 + DR-7 match rule.

Seed rule (derived from the archived rows — verified on all 270 — and from
scripts/real_logistic_gpu.py lines 189/209-210):
  seed        = seed_base{wdbc:610000, spambase:620000, adult:630000}
                + 10000*replicate + prior_idx        (prior_idx: 0.1->0, 0.3->1)
  design_seed = seed_base + prior_idx
Consumed streams (harness construction): method seed+101/+202/+303
(iid/zero_sum/antithetic), reference seed+77717, probe design_seed+1729
(replicate-independent, AM-9).

Note for this repository: the fresh-mode inputs and the protocol documents
named above are not included.  Archived mode needs only the files in this
repository.  Host names are redacted in the released rows, so host checks
compare against the redaction token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_TEX = REPO_ROOT / "paper" / "tables" / "real_odd_k_summary.tex"
ANALYZER = REPO_ROOT / "campaigns" / "analyze_attempt02.py"
ARCHIVED_DIR_DEFAULT = REPO_ROOT / "campaigns" / "attempt02"
FRESH_DIR_DEFAULT = REPO_ROOT / "outputs" / "sweeps" / "camera_ready_regen" / "oddk"
FRESH_TABLES_DEFAULT = FRESH_DIR_DEFAULT / "tables"
SCRATCH_DEFAULT = Path("/tmp/regen_g2")

DATASETS = ("wdbc", "spambase", "adult")
DISPLAY = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}
PRIORS = (0.1, 0.3)
PRIOR_STR = {0.1: "0.1", 0.3: "0.3"}
METHODS = ("iid", "zero_sum", "antithetic")

SEED_BASE = {"wdbc": 610000, "spambase": 620000, "adult": 630000}
METHOD_OFFSET = {"iid": 101, "zero_sum": 202, "antithetic": 303}
REF_OFFSET = 77717
PROBE_OFFSET = 1729
REP_SHIFT = 40032  # AM-1b: fresh replicate = archived replicate + 40032
REGISTERED_FRESH_ENVELOPE = (401_020_000, 401_257_718)  # AM-1b

DATASET_PIN = {
    "wdbc": dict(n_ensembles=256, dim=31, steps=900, burnin=450, windows=450,
                 ref_steps=1800, ref_burnin=900),
    "spambase": dict(n_ensembles=192, dim=58, steps=900, burnin=450, windows=450,
                     ref_steps=1800, ref_burnin=900),
    "adult": dict(n_ensembles=64, dim=105, steps=700, burnin=350, windows=350,
                  ref_steps=1400, ref_burnin=700),
}
ETA_PIN = {0.1: 0.08, 0.3: 0.054545454545454536}
REF_REPS_PIN = 16
REF_CHAINS_PIN = 16
GIT_HASH_ARCHIVED = "184ea65"

# DR-4a / VR-3: exact per-row key set ENUMERATED FROM THE ARCHIVED RAWS
# (all 270 rows carry exactly these 38 keys); ODDK fresh pin == archived pin
# (harness byte-identical to its June state).
STR_KEYS = frozenset({"coupling", "dataset", "git_hash", "gpu", "method", "target"})
INT_KEYS = frozenset({"burnin", "chains", "design_seed", "dim", "n_ensembles",
                      "ref_burnin", "ref_chains", "ref_reps", "ref_steps",
                      "replicate", "reps", "seed", "steps", "windows"})
FLOAT_KEYS = frozenset({"bias_sq", "elapsed_sec", "elapsed_sec_per_ensemble", "eta",
                        "lin_chain_var", "lin_ens_var", "marginal_second", "mse",
                        "mse_to_single", "prior", "single_bias_sq", "single_mse",
                        "single_test_logloss", "single_trace_var", "terminal_spread",
                        "test_logloss", "test_logloss_var", "trace_var"})
KEYSET_PIN = STR_KEYS | INT_KEYS | FLOAT_KEYS
assert len(KEYSET_PIN) == 38

BLOCKS = ("primary", "stress", "edge")
BLOCK_PREFIX = {"primary": "primary_rep", "stress": "stress_k7_rep", "edge": "edge_k5_rep"}
BLOCK_CHAINS = {"primary": 7, "stress": 7, "edge": 5}
BLOCK_ROWS = {"primary": 108, "stress": 108, "edge": 54}
REPLICATES = {
    "archived": {"primary": tuple(range(9, 15)), "stress": tuple(range(15, 21)),
                 "edge": tuple(range(21, 24))},
    "fresh": {"primary": tuple(range(40041, 40047)), "stress": tuple(range(40047, 40053)),
              "edge": tuple(range(40053, 40056))},
}

HEADER_LINE = (r"Dataset & prior & $n_{\rm ens}$ & $K=7$ $C_\star$/pair+1 primary & "
               r"$K=7$ $C_\star$/pair+1 stress & pair+1/iid primary;stress & "
               r"$K=5$ $C_\star$/pair+1 edge & pair+1/iid edge \\")
ROW_SUFFIX = " " + "\\" * 2

# cell layout: (key, block, numerator, denominator, kind)
CELL_STATS = (
    ("primary_zero_over_ant", "primary", "zero_sum", "antithetic", "ranged"),
    ("stress_zero_over_ant", "stress", "zero_sum", "antithetic", "ranged"),
    ("primary_ant_over_iid", "primary", "antithetic", "iid", "scalar"),
    ("stress_ant_over_iid", "stress", "antithetic", "iid", "scalar"),
    ("edge_zero_over_ant", "edge", "zero_sum", "antithetic", "ranged"),
    ("edge_ant_over_iid", "edge", "antithetic", "iid", "scalar"),
)
COLUMN_NAME = {
    "primary_zero_over_ant": "$K=7$ $C_\\star$/pair+1 primary",
    "stress_zero_over_ant": "$K=7$ $C_\\star$/pair+1 stress",
    "primary_ant_over_iid": "pair+1/iid primary;stress (primary median)",
    "stress_ant_over_iid": "pair+1/iid primary;stress (stress median)",
    "edge_zero_over_ant": "$K=5$ $C_\\star$/pair+1 edge",
    "edge_ant_over_iid": "pair+1/iid edge",
}

RANGED_RE = re.compile(r"^(\d+\.\d{4}) \[(\d+\.\d{4}), (\d+\.\d{4})\]$")
PAIR_RE = re.compile(r"^(\d+\.\d{4}) / (\d+\.\d{4})$")
SCALAR_RE = re.compile(r"^(\d+\.\d{4})$")


def f4(value):
    return f"{value:.4f}"


def fmt_analyzer(value):
    """Byte replica of analyze_attempt02.py fmt() for the cross-check."""
    if value is None or not math.isfinite(value):
        return "--"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def median(values):
    """np.median-equivalent for float64 lists (even n: mean of middle two)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def abort(errors, phase):
    if errors:
        sys.stderr.write(f"FAIL-CLOSED [{phase}]: {len(errors)} validation error(s); "
                         "no output written; exiting nonzero.\n")
        for err in errors[:40]:
            sys.stderr.write("  - " + err + "\n")
        if len(errors) > 40:
            sys.stderr.write(f"  ... and {len(errors) - 40} more\n")
        sys.exit(1)


def expected_seed(dataset, replicate, prior):
    return SEED_BASE[dataset] + 10000 * replicate + PRIORS.index(prior)


def expected_design_seed(dataset, prior):
    return SEED_BASE[dataset] + PRIORS.index(prior)


def discover_files(input_dir, mode, errors):
    """Discover the 15 per-block input files; pin exact filename/replicate sets."""
    expected = {}
    for block in BLOCKS:
        for rep in REPLICATES[mode][block]:
            expected[(block, rep)] = input_dir / f"{BLOCK_PREFIX[block]}{rep}.jsonl"
    if not input_dir.is_dir():
        errors.append(f"input dir does not exist: {input_dir}")
        return expected
    expected_names = {p.name for p in expected.values()}
    found_names = {p.name for p in sorted(input_dir.glob("*.jsonl"))}
    unexpected = sorted(found_names - expected_names)
    missing = sorted(expected_names - found_names)
    if unexpected:
        errors.append(f"unexpected .jsonl files in {input_dir} for mode={mode} "
                      f"(exact {mode} replicate sets "
                      f"{ {b: list(REPLICATES[mode][b]) for b in BLOCKS} } are pinned): "
                      + ", ".join(unexpected))
    if missing:
        errors.append(f"missing expected {mode} input files in {input_dir}: "
                      + ", ".join(missing))
    return expected


def validate_file(path, block, rep, mode, errors, trace_vars):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path.name}: unreadable ({exc})")
        return 0
    rows = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append((lineno, json.loads(line)))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{lineno}: invalid JSON ({exc})")
    if len(rows) != 18:
        errors.append(f"{path.name}: expected 18 rows, found {len(rows)}")
    seen = set()
    for lineno, row in rows:
        ctx = f"{path.name}:{lineno}"
        keys = set(row)
        if keys != KEYSET_PIN:
            extra = sorted(keys - KEYSET_PIN)
            miss = sorted(KEYSET_PIN - keys)
            errors.append(f"{ctx}: key set != 38-key archived pin"
                          f"{' extra=' + ','.join(extra) if extra else ''}"
                          f"{' missing=' + ','.join(miss) if miss else ''}")
            continue
        bad_type = False
        for key, value in row.items():
            if key in STR_KEYS:
                if not isinstance(value, str) or not value:
                    errors.append(f"{ctx}: {key} must be nonempty str, got {value!r}")
                    bad_type = True
            elif key in INT_KEYS:
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"{ctx}: {key} must be int, got {value!r}")
                    bad_type = True
            else:
                if not isinstance(value, float) or not math.isfinite(value):
                    errors.append(f"{ctx}: {key} must be finite float, got {value!r}")
                    bad_type = True
        if bad_type:
            continue
        dataset, prior, method = row["dataset"], row["prior"], row["method"]
        if row["target"] != "real_logistic":
            errors.append(f"{ctx}: target={row['target']!r} != 'real_logistic'")
            continue
        if dataset not in DATASETS or prior not in PRIORS or method not in METHODS:
            errors.append(f"{ctx}: unknown dataset/prior/method "
                          f"({dataset!r}, {prior!r}, {method!r})")
            continue
        if row["coupling"] != method:
            errors.append(f"{ctx}: coupling={row['coupling']!r} != method={method!r}")
        if row["chains"] != BLOCK_CHAINS[block]:
            errors.append(f"{ctx}: chains={row['chains']} != {BLOCK_CHAINS[block]} "
                          f"for block {block}")
        pin = DATASET_PIN[dataset]
        for key in ("n_ensembles", "dim", "steps", "burnin", "windows",
                    "ref_steps", "ref_burnin"):
            if row[key] != pin[key]:
                errors.append(f"{ctx}: {key}={row[key]} != pinned {pin[key]} for {dataset}")
        if row["reps"] != pin["n_ensembles"]:
            errors.append(f"{ctx}: reps={row['reps']} != n_ensembles pin {pin['n_ensembles']}")
        if row["windows"] != row["steps"] - row["burnin"]:
            errors.append(f"{ctx}: windows != steps - burnin")
        if row["ref_steps"] != 2 * row["steps"] or row["ref_burnin"] != 2 * row["burnin"]:
            errors.append(f"{ctx}: reference horizon != 2x (ref_steps/ref_burnin)")
        if row["ref_reps"] != REF_REPS_PIN or row["ref_chains"] != REF_CHAINS_PIN:
            errors.append(f"{ctx}: ref_reps/ref_chains != {REF_REPS_PIN}/{REF_CHAINS_PIN}")
        if row["eta"] != ETA_PIN[prior]:
            errors.append(f"{ctx}: eta={row['eta']!r} != pinned {ETA_PIN[prior]!r}")
        if row["replicate"] != rep:
            errors.append(f"{ctx}: replicate={row['replicate']} != file replicate {rep}")
        seed_want = expected_seed(dataset, rep, prior)
        if row["seed"] != seed_want:
            errors.append(f"{ctx}: seed={row['seed']} != exact expected {seed_want} "
                          f"(= {SEED_BASE[dataset]} + 10000*{rep} + {PRIORS.index(prior)}; DR-4d)")
        dseed_want = expected_design_seed(dataset, prior)
        if row["design_seed"] != dseed_want:
            errors.append(f"{ctx}: design_seed={row['design_seed']} != expected {dseed_want}")
        if mode == "archived" and row["git_hash"] != GIT_HASH_ARCHIVED:
            errors.append(f"{ctx}: git_hash={row['git_hash']!r} != archived pin "
                          f"{GIT_HASH_ARCHIVED!r}")
        if row["trace_var"] <= 0.0:
            errors.append(f"{ctx}: trace_var={row['trace_var']!r} not > 0")
        combo = (dataset, prior, method)
        if combo in seen:
            errors.append(f"{ctx}: duplicate (dataset, prior, method) {combo}")
            continue
        seen.add(combo)
        trace_vars[(block, dataset, prior, rep, method)] = row["trace_var"]
    full = {(d, p, m) for d in DATASETS for p in PRIORS for m in METHODS}
    if seen != full:
        errors.append(f"{path.name}: incomplete method battery; missing "
                      f"{sorted(full - seen)}")
    return len(rows)


def load_and_validate(input_dir, mode, phase, verbose=True):
    errors = []
    files = discover_files(input_dir, mode, errors)
    abort(errors, phase)
    trace_vars = {}
    block_counts = defaultdict(int)
    for (block, rep), path in sorted(files.items()):
        block_counts[block] += validate_file(path, block, rep, mode, errors, trace_vars)
    for block in BLOCKS:
        if block_counts[block] != BLOCK_ROWS[block]:
            errors.append(f"block {block}: {block_counts[block]} rows != pinned "
                          f"{BLOCK_ROWS[block]}")
    total = sum(block_counts.values())
    if total != 270:
        errors.append(f"grand total {total} rows != 270")
    abort(errors, phase)
    if verbose:
        print(f"[{phase}] validated {total} rows = "
              + " + ".join(f"{block_counts[b]} {b}" for b in BLOCKS)
              + f"; 15 files; key-set pin: 38 keys/row; exact per-row seed/design_seed "
              f"recomputation PASS (mode={mode})")
    return files, trace_vars


def compute_cells(trace_vars, mode):
    """Per cell (dataset, prior): per-replicate ratios -> median/min/max per stat."""
    cells = {}
    for dataset in DATASETS:
        for prior in PRIORS:
            cell = {}
            for key, block, num, den, kind in CELL_STATS:
                per_rep = {}
                for rep in REPLICATES[mode][block]:
                    per_rep[rep] = (trace_vars[(block, dataset, prior, rep, num)]
                                    / trace_vars[(block, dataset, prior, rep, den)])
                vals = list(per_rep.values())
                cell[key] = {"kind": kind, "median": median(vals), "min": min(vals),
                             "max": max(vals),
                             "per_replicate": {str(r): per_rep[r] for r in sorted(per_rep)}}
            cells[(dataset, prior)] = cell
    return cells


def ranged_str(stat):
    return f"{f4(stat['median'])} [{f4(stat['min'])}, {f4(stat['max'])}]"


def render_tex(cells):
    lines = [r"\begin{tabular}{llclllll}", r"\toprule", HEADER_LINE, r"\midrule"]
    for dataset in DATASETS:
        for prior in PRIORS:
            cell = cells[(dataset, prior)]
            parts = [
                DISPLAY[dataset],
                PRIOR_STR[prior],
                str(DATASET_PIN[dataset]["n_ensembles"]),
                ranged_str(cell["primary_zero_over_ant"]),
                ranged_str(cell["stress_zero_over_ant"]),
                f4(cell["primary_ant_over_iid"]["median"]) + " / "
                + f4(cell["stress_ant_over_iid"]["median"]),
                ranged_str(cell["edge_zero_over_ant"]),
                f4(cell["edge_ant_over_iid"]["median"]),
            ]
            lines.append(" & ".join(parts) + ROW_SUFFIX)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def reparse_check(tex_text, cells, phase):
    """Structural self-check: re-parse the emitted tabular; string round-trip."""
    errors = []
    lines = tex_text.splitlines()
    frame = [r"\begin{tabular}{llclllll}", r"\toprule", HEADER_LINE, r"\midrule"]
    if len(lines) != 12 or lines[:4] != frame or lines[10] != r"\bottomrule" \
            or lines[11] != r"\end{tabular}" or not tex_text.endswith("\n"):
        errors.append("emitted .tex frame does not match the frozen 12-line structure")
        abort(errors, phase)
    n_ranged = n_scalar = 0
    expected_rows = [(d, p) for d in DATASETS for p in PRIORS]
    for line, (dataset, prior) in zip(lines[4:10], expected_rows):
        if not line.endswith(ROW_SUFFIX):
            errors.append(f"data row missing ' \\\\' suffix: {line!r}")
            continue
        cols = line[:-len(ROW_SUFFIX)].split(" & ")
        if len(cols) != 8:
            errors.append(f"data row does not split into 8 columns: {line!r}")
            continue
        cell = cells[(dataset, prior)]
        if cols[0] != DISPLAY[dataset] or cols[1] != PRIOR_STR[prior] \
                or cols[2] != str(DATASET_PIN[dataset]["n_ensembles"]):
            errors.append(f"row identity columns wrong for ({dataset}, {prior}): {cols[:3]}")
        for col, key in ((3, "primary_zero_over_ant"), (4, "stress_zero_over_ant"),
                         (6, "edge_zero_over_ant")):
            if not RANGED_RE.match(cols[col]):
                errors.append(f"({dataset},{prior}) col {col} not a ranged cell: {cols[col]!r}")
            elif cols[col] != ranged_str(cell[key]):
                errors.append(f"({dataset},{prior}) col {col} round-trip mismatch")
            else:
                n_ranged += 1
        if not PAIR_RE.match(cols[5]):
            errors.append(f"({dataset},{prior}) pair column malformed: {cols[5]!r}")
        elif cols[5] != (f4(cell["primary_ant_over_iid"]["median"]) + " / "
                         + f4(cell["stress_ant_over_iid"]["median"])):
            errors.append(f"({dataset},{prior}) pair column round-trip mismatch")
        else:
            n_scalar += 2
        if not SCALAR_RE.match(cols[7]):
            errors.append(f"({dataset},{prior}) edge scalar malformed: {cols[7]!r}")
        elif cols[7] != f4(cell["edge_ant_over_iid"]["median"]):
            errors.append(f"({dataset},{prior}) edge scalar round-trip mismatch")
        else:
            n_scalar += 1
    if n_ranged != 18:
        errors.append(f"ranged-cell count {n_ranged} != 18")
    if n_scalar != 18:
        errors.append(f"scalar-median count {n_scalar} != 18")
    abort(errors, phase)
    print(f"[{phase}] self-check: emitted tabular re-parsed OK; "
          f"{n_ranged} ranged cells + {n_scalar} scalar medians; string round-trip OK")


def cross_check(files, cells, mode, scratch_dir, phase):
    """Cross-check all 36 medians against campaigns/analyze_attempt02.py."""
    errors = []
    block_map = {
        "primary": ("primary_zero_over_ant", "primary_ant_over_iid"),
        "stress": ("stress_zero_over_ant", "stress_ant_over_iid"),
        "edge": ("edge_zero_over_ant", "edge_ant_over_iid"),
    }
    n_medians = n_repsets = 0
    for block in BLOCKS:
        paths = [str(files[(block, rep)]) for rep in REPLICATES[mode][block]]
        out_md = scratch_dir / f"analyze_xcheck_{phase.replace(':', '_')}_{block}.md"
        cmd = [sys.executable, str(ANALYZER), *paths,
               "--chains", str(BLOCK_CHAINS[block]), "--out", str(out_md)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            errors.append(f"analyze_attempt02.py failed for block {block}: "
                          f"{proc.stderr.strip()[:300]}")
            continue
        table = {}
        for line in out_md.read_text(encoding="utf-8").splitlines():
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 11 and cols[1] in DATASETS:
                # | dataset | prior | reps | zero tr | ant tr | iid tr | z/a | z/i | a/i |...
                table[(cols[1], cols[2])] = (cols[3], cols[7], cols[9])
        expect_reps = ",".join(str(r) for r in REPLICATES[mode][block])
        for dataset in DATASETS:
            for prior in PRIORS:
                got = table.get((dataset, f"{prior:.1f}"))
                if got is None:
                    errors.append(f"analyzer output missing cell ({dataset}, {prior}) "
                                  f"in block {block}")
                    continue
                reps_str, za_str, ai_str = got
                if reps_str != expect_reps:
                    errors.append(f"analyzer matched reps {reps_str!r} != expected "
                                  f"{expect_reps!r} ({block}, {dataset}, {prior})")
                else:
                    n_repsets += 1
                za_key, ai_key = block_map[block]
                cell = cells[(dataset, prior)]
                for key, got_str in ((za_key, za_str), (ai_key, ai_str)):
                    want = fmt_analyzer(cell[key]["median"])
                    if want != got_str:
                        errors.append(f"median mismatch vs analyzer ({block}, {dataset}, "
                                      f"{prior}, {key}): generator {want} != analyzer {got_str}")
                    else:
                        n_medians += 1
    abort(errors, phase)
    print(f"[{phase}] cross-check vs campaigns/analyze_attempt02.py: "
          f"{n_medians}/36 medians agree; {n_repsets}/18 replicate sets agree")


def safe_write(path, data, phase):
    """Write bytes; hard-refuse anything under <repo>/paper; verify write-back."""
    path = Path(path).resolve()
    paper_dir = (REPO_ROOT / "paper").resolve()
    if path == paper_dir or paper_dir in path.parents:
        abort([f"refusing to write under {paper_dir}: {path} (DR-4e)"], phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if path.read_bytes() != data:
        abort([f"write-back verification failed for {path}"], phase)
    return path


def archived_gate(input_dir, scratch_dir, no_cross_check, phase, write_artifact):
    """Validate archived raws, render, byte-compare to the frozen committed .tex."""
    files, trace_vars = load_and_validate(input_dir, "archived", phase)
    cells = compute_cells(trace_vars, "archived")
    tex_text = render_tex(cells)
    emitted = tex_text.encode("utf-8")
    frozen = FROZEN_TEX.read_bytes()
    if emitted != frozen:
        errs = [f"emitted .tex is NOT byte-identical to {FROZEN_TEX} "
                f"(emitted {len(emitted)} bytes sha256={sha256_bytes(emitted)}, "
                f"frozen {len(frozen)} bytes sha256={sha256_bytes(frozen)})"]
        for i, (got, want) in enumerate(zip(tex_text.splitlines(),
                                            frozen.decode("utf-8").splitlines()), 1):
            if got != want:
                errs.append(f"first differing line {i}: emitted={got!r} frozen={want!r}")
                break
        abort(errs, phase)
    print("ARCHIVED GATE PASS (byte-identical)")
    print(f"  frozen : {FROZEN_TEX}  sha256={sha256_bytes(frozen)}")
    artifact = None
    if write_artifact:
        artifact = safe_write(scratch_dir / "real_odd_k_summary_archived_regen.tex",
                              emitted, phase)
        print(f"  emitted: {artifact}  sha256={sha256_bytes(emitted)} (identical)")
    else:
        print(f"  emitted: (in-memory)  sha256={sha256_bytes(emitted)} (identical)")
    reparse_check(tex_text, cells, phase)
    if not no_cross_check:
        cross_check(files, cells, "archived", scratch_dir, phase)
    return files, cells, tex_text


def frozen_printed_strings(stat, kind):
    if kind == "ranged":
        return ranged_str(stat)
    return f4(stat["median"])


def build_comparison(cells_frozen, cells_fresh, files_archived, files_fresh,
                     fresh_tex_bytes, argv):
    ranged_pass = scalar_pass = 0
    cell_records = []
    for dataset in DATASETS:
        for prior in PRIORS:
            for key, block, num, den, kind in CELL_STATS:
                fz = cells_frozen[(dataset, prior)][key]
                fr = cells_fresh[(dataset, prior)][key]
                frozen_full = fz["median"]
                rec = {
                    "cell_id": f"{dataset}/{PRIOR_STR[prior]}/{key}",
                    "dataset": dataset,
                    "prior": PRIOR_STR[prior],
                    "column": COLUMN_NAME[key],
                    "block": block,
                    "ratio": f"trace_var({num})/trace_var({den}) at K={BLOCK_CHAINS[block]}",
                    "kind": kind,
                    "frozen_printed": frozen_printed_strings(fz, kind),
                    "frozen_full_precision": (
                        {"median": fz["median"], "min": fz["min"], "max": fz["max"]}
                        if kind == "ranged" else {"median": fz["median"]}),
                    "fresh": {"median": fr["median"], "min": fr["min"], "max": fr["max"],
                              "per_replicate": fr["per_replicate"]},
                }
                within = fr["min"] <= frozen_full <= fr["max"]
                if kind == "ranged":
                    rec["clause_used"] = ("ranged: frozen full-precision median within "
                                          "fresh per-replicate [min, max] (AM-13/DR-7)")
                    rec["verdict"] = "PASS" if within else "FAIL"
                    ranged_pass += within
                else:
                    if frozen_full == 0.0:
                        rec["clause_used"] = "15pct rule inapplicable (zero frozen; DR-7b)"
                        rec["verdict"] = "FAIL"
                    else:
                        rel = abs(fr["median"] - frozen_full) / abs(frozen_full)
                        rec["clause_used"] = (
                            "scalar: frozen full-precision median within fresh min-max of "
                            "the underlying per-replicate values (AM-13 primary clause; "
                            "per-replicate values exist for every ODDK scalar cell)")
                        rec["verdict"] = "PASS" if within else "FAIL"
                        rec["relative_deviation"] = rel
                        rec["fifteen_pct_rule"] = {
                            "formula": "|fresh_median - frozen| / |frozen| <= 0.15 (DR-7b)",
                            "value": rel,
                            "would_pass": rel <= 0.15,
                            "status": ("informational: operative under AM-13 only where "
                                       "per-replicate values do not exist"),
                        }
                        scalar_pass += within
                cell_records.append(rec)
    disclosure = seed_sharing_disclosure()
    record = {
        "schema": "regen-comparison-1",
        "campaign": "oddk",
        "table": "real_odd_k_summary",
        "generator": "scripts/regen_oddk_table.py",
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "invocation_argv": list(argv),
        "spec_citations": [
            "campaigns/camera_ready_regen_invocation_plan.md SECTION 2 (Revision 5) + '### G2'",
            "campaigns/camera_ready_regen_precommit.md AM-1b, AM-9, AM-13, DR-4(a-e), DR-7(a,b), VR-2, VR-3",
        ],
        "layer_framing": "AM-0: the archived-mode byte-identity gate is the RESTORATION layer; this record is the CONFIRMATION layer (fresh disjoint seeds; frozen tables remain the printed audited objects this cycle)",
        "frozen_tex": {"path": str(FROZEN_TEX.relative_to(REPO_ROOT)),
                       "sha256": sha256_file(FROZEN_TEX)},
        "archived_gate": "PASS (byte-identical reproduction of the frozen .tex from the "
                         "archived raws in this same invocation, before fresh raws were trusted)",
        "fresh_tex_sha256": sha256_bytes(fresh_tex_bytes),
        "inputs": {
            "archived": {p.name: sha256_file(p) for _, p in sorted(files_archived.items())},
            "fresh": {p.name: sha256_file(p) for _, p in sorted(files_fresh.items())},
        },
        "replicate_map": {
            "archived": {b: list(REPLICATES["archived"][b]) for b in BLOCKS},
            "fresh": {b: list(REPLICATES["fresh"][b]) for b in BLOCKS},
            "rule": f"fresh = archived + {REP_SHIFT} (AM-1b)",
        },
        "match_rule": {
            "ranged_cells": "frozen full-precision median within fresh per-replicate "
                            "[min, max] (AM-13; DR-7a full-precision comparanda with the "
                            "rounded printed value recorded alongside)",
            "scalar_cells": "frozen full-precision median within fresh min-max of the "
                            "underlying per-replicate values where such exist (they exist "
                            "for all 18 ODDK scalar medians), else 15% relative "
                            "|fresh_median - frozen|/|frozen| <= 0.15 (DR-7b); the 15% "
                            "figure is additionally recorded informationally per cell",
        },
        "cells": cell_records,
        "summary": {
            "n_cells": len(cell_records),
            "n_ranged": 18,
            "n_scalar": 18,
            "ranged_pass": ranged_pass,
            "scalar_pass": scalar_pass,
            "pass": ranged_pass + scalar_pass,
            "fail": len(cell_records) - ranged_pass - scalar_pass,
            "note": "pass fractions are DESCRIPTIVE under the registered match rule only, "
                    "never calibrated statistics (DR-8); min-max coverage differs between "
                    "3- and 6-replicate cells",
        },
        "seed_sharing_disclosure": disclosure,
    }
    return record


def seed_sharing_disclosure():
    """VR-2: disclose intentional same-design sharing + accidental cross-target
    coincidences for the fresh map; verify the registered AM-1b envelope."""
    entries = defaultdict(list)
    bases = []
    for block in BLOCKS:
        for rep in REPLICATES["fresh"][block]:
            for dataset in DATASETS:
                for prior in PRIORS:
                    base = expected_seed(dataset, rep, prior)
                    bases.append(base)
                    entries[base].append((dataset, rep, PRIOR_STR[prior], "base"))
                    for method, off in METHOD_OFFSET.items():
                        entries[base + off].append(
                            (dataset, rep, PRIOR_STR[prior], f"method_{method}"))
                    entries[base + REF_OFFSET].append(
                        (dataset, rep, PRIOR_STR[prior], "reference"))
    cross_target = {}
    same_target = {}
    for seed, users in entries.items():
        datasets_used = {u[0] for u in users}
        if len(datasets_used) > 1:
            cross_target[seed] = users
        for dataset in datasets_used:
            mine = [u for u in users if u[0] == dataset]
            if len({(u[1], u[2]) for u in mine}) > 1:
                same_target[seed] = mine
    envelope = (min(bases), max(bases) + REF_OFFSET)
    if envelope != REGISTERED_FRESH_ENVELOPE:
        abort([f"fresh consumed-seed envelope {envelope} != registered AM-1b "
               f"{REGISTERED_FRESH_ENVELOPE}"], "fresh:envelope")
    examples = []
    for seed in sorted(cross_target)[:3]:
        users = "; ".join(f"{d} rep {r} prior {p} [{s}]" for d, r, p, s in
                          sorted(cross_target[seed]))
        examples.append(f"{seed}: {users}")
    return {
        "rule": "VR-2: no blanket distinctness claim; the authoritative property is "
                "exact per-row seed recomputation (DR-4d, enforced by this generator); "
                "sharing falls in two disclosed categories",
        "intentional_same_design": [
            "within each cell (dataset, replicate, prior) the three method rows carry the "
            "same base seed (row field `seed`); method streams consume seed+101/+202/+303 "
            "(iid/zero_sum/antithetic)",
            "the reference stream seed+77717 is shared across the three methods of a cell "
            "(matched reference by construction)",
            "probe stream design_seed+1729 is replicate-independent (AM-9), outside the "
            "401M block, and feeds only lin_* diagnostic side fields",
            f"example: wdbc rep 40041 prior 0.1 -> base {expected_seed('wdbc', 40041, 0.1)}, "
            f"shared reference stream {expected_seed('wdbc', 40041, 0.1) + REF_OFFSET}",
        ],
        "accidental_cross_target": {
            "mechanism": "dataset seed bases 610000/620000/630000 sit 10000 apart = one "
                         "replicate step, so adjacent-dataset/adjacent-replicate seeds "
                         "coincide numerically across targets (DR-3/AM-10(ii)); different "
                         "targets/designs, streams diverge immediately",
            "n_coincident_seed_values": len(cross_target),
            "examples": examples,
        },
        "same_target_cross_replicate_nonprobe_collisions": {
            "count": len(same_target),
            "statement": "computed over all fresh base/method/reference seeds; expected 0 "
                         "(VR-2 round-2 enumeration found none)",
        },
        "consumed_envelope": {
            "min_base": envelope[0],
            "max_consumed": envelope[1],
            "registered_am1b": list(REGISTERED_FRESH_ENVELOPE),
            "inside_subblock_401.0M_401.9M": True,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="G2 generator for paper/tables/real_odd_k_summary.tex "
                    "(camera-ready regeneration campaign; fail-closed)")
    parser.add_argument("--mode", required=True, choices=("archived", "fresh"))
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="raw JSONL dir (default: campaigns/attempt02 for "
                             "archived; outputs/sweeps/camera_ready_regen/oddk for fresh)")
    parser.add_argument("--archived-dir", type=Path, default=ARCHIVED_DIR_DEFAULT,
                        help="archived raws for the fresh-mode trust gate + frozen "
                             "full-precision comparanda (DR-7a)")
    parser.add_argument("--out-dir", type=Path, default=FRESH_TABLES_DEFAULT,
                        help="fresh-mode output dir (registered default per DR-4e; "
                             "override is test-only and never under paper/)")
    parser.add_argument("--scratch-dir", type=Path, default=SCRATCH_DEFAULT,
                        help="scratch dir for the archived gate artifact + cross-check md")
    parser.add_argument("--no-cross-check", action="store_true",
                        help="skip the analyze_attempt02.py median cross-check")
    args = parser.parse_args(argv)

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "archived":
        input_dir = args.input_dir or ARCHIVED_DIR_DEFAULT
        archived_gate(input_dir, args.scratch_dir, args.no_cross_check,
                      "archived", write_artifact=True)
        return 0

    # fresh mode: dual-mode trust rule — the archived gate must pass in this
    # same invocation before any fresh raw is trusted (SECTION 2 common reqs).
    _, cells_frozen, _ = archived_gate(args.archived_dir, args.scratch_dir,
                                       args.no_cross_check, "fresh:archived-gate",
                                       write_artifact=False)
    input_dir = args.input_dir or FRESH_DIR_DEFAULT
    files_fresh, trace_vars = load_and_validate(input_dir, "fresh", "fresh")
    cells_fresh = compute_cells(trace_vars, "fresh")
    fresh_tex = render_tex(cells_fresh)
    reparse_check(fresh_tex, cells_fresh, "fresh")
    if not args.no_cross_check:
        cross_check(files_fresh, cells_fresh, "fresh", args.scratch_dir, "fresh")
    files_archived = discover_files(args.archived_dir, "archived", [])
    record = build_comparison(cells_frozen, cells_fresh, files_archived, files_fresh,
                              fresh_tex.encode("utf-8"), sys.argv)
    record_bytes = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode("utf-8")
    tex_path = safe_write(args.out_dir / "real_odd_k_summary_fresh.tex",
                          fresh_tex.encode("utf-8"), "fresh:write")
    json_path = safe_write(args.out_dir / "comparison_oddk.json", record_bytes,
                           "fresh:write")
    summary = record["summary"]
    print(f"[fresh] wrote {tex_path}  sha256={record['fresh_tex_sha256']}")
    print(f"[fresh] wrote {json_path}  sha256={sha256_bytes(record_bytes)}")
    print(f"[fresh] match-rule verdicts (descriptive, DR-8): "
          f"{summary['pass']}/{summary['n_cells']} PASS "
          f"({summary['ranged_pass']}/18 ranged, {summary['scalar_pass']}/18 scalar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
