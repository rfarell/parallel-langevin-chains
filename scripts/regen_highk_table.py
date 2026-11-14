#!/usr/bin/env python
"""G1 deterministic regenerator for paper/tables/campaign_highk_iid.tex.

Registered spec: campaigns/camera_ready_regen_invocation_plan.md
SECTION 2 common requirements (Revision 5) + "### G1"; precommit rulings
DR-4(a-e), DR-7(a-c), VR-3, protocol item 5 (match-or-disclose) in
campaigns/camera_ready_regen_precommit.md.  CPU-only, stdlib-only,
deterministic; fail-closed: any validator/gate failure prints the failure,
writes NO output, exits nonzero.

Modes:
  archived  Gate mode (DR-4b): recompute the table from the archived June
            raws (campaigns/attempt01/) and require BYTE-IDENTICAL
            equality with the committed paper/tables/campaign_highk_iid.tex.
            Optional --emit writes the emitted bytes OUTSIDE paper/ for
            inspection; paper/tables/ is never written.
  fresh     Confirmation mode: validate the registered fresh raws
            (replicate map 39939-39941 primary / 39942-39944 stress; exact
            per-row seed/design_seed recomputation per DR-4d; key pin equal
            to the archived 38-key enumeration per VR-3), re-run the
            archived byte gate in memory to reconstruct full-precision
            frozen comparanda (DR-7a), then emit
              outputs/sweeps/camera_ready_regen/highk/tables/
                campaign_highk_iid_fresh.tex
                comparison_highk.json
            implementing protocol item 5 + AM-13 + DR-7b/c (the two
            paper-quoted scalars (Results paragraph) get BOTH clauses).  DR-4e: fresh outputs
            go ONLY under outputs/sweeps/camera_ready_regen/highk/tables/.

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
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN_TEX = REPO / "paper" / "tables" / "campaign_highk_iid.tex"
ARCHIVED_DIR = REPO / "campaigns" / "attempt01"
FRESH_DIR = REPO / "outputs" / "sweeps" / "camera_ready_regen" / "highk"
FRESH_TABLES_DIR = FRESH_DIR / "tables"

# Exact 38-key schema, ENUMERATED from the archived attempt01 rows (DR-4a;
# all 126 rows carry exactly this set).  VR-3: HIGHK fresh pin == archived
# pin (harness scripts/real_logistic_gpu.py byte-identical to June state).
KEY_PIN = frozenset([
    "bias_sq", "burnin", "chains", "coupling", "dataset", "design_seed",
    "dim", "elapsed_sec", "elapsed_sec_per_ensemble", "eta", "git_hash",
    "gpu", "lin_chain_var", "lin_ens_var", "marginal_second", "method",
    "mse", "mse_to_single", "n_ensembles", "prior", "ref_burnin",
    "ref_chains", "ref_reps", "ref_steps", "replicate", "reps", "seed",
    "single_bias_sq", "single_mse", "single_test_logloss",
    "single_trace_var", "steps", "target", "terminal_spread",
    "test_logloss", "test_logloss_var", "trace_var", "windows",
])

# Harness constants (scripts/real_logistic_gpu.py: seed_base map, seed =
# base + 10000*replicate + idx(prior), design_seed = base + idx(prior);
# verified against all 126 archived rows with zero mismatches).
SEED_BASE = {"wdbc": 610000, "spambase": 620000, "adult": 630000}
ETA_BY_PRIOR = {0: 0.08, 1: 0.054545454545454536}
# dataset -> (label, reps@K=8, dim, steps, burnin, ref_steps, ref_burnin)
DATASET_PIN = {
    "wdbc": ("WDBC", 256, 31, 900, 450, 1800, 900),
    "spambase": ("Spambase", 192, 58, 900, 450, 1800, 900),
    "adult": ("Adult", 64, 105, 700, 350, 1400, 700),
}
DATASET_ORDER = ["wdbc", "spambase", "adult"]
PRIORS = [(0.1, "0.1"), (0.3, "0.3")]
K128_REPS = 24
PRIMARY_SET = frozenset([("zero_sum", 8), ("antithetic", 8), ("iid", 128)])
STRESS_SET = frozenset(
    [("iid", 8), ("zero_sum", 8), ("antithetic", 8), ("iid", 128)])

REP_MAP = {
    "archived": {"primary": (3, 4, 5), "stress": (6, 7, 8)},
    "fresh": {"primary": (39939, 39940, 39941),
              "stress": (39942, 39943, 39944)},
}
ARCHIVED_FILES = {
    "primary": [ARCHIVED_DIR / ("rep%d.jsonl" % r)
                for r in REP_MAP["archived"]["primary"]],
    "stress": [ARCHIVED_DIR / ("stress_rep%d.jsonl" % r)
               for r in REP_MAP["archived"]["stress"]],
}
FRESH_FILES = {
    "primary": [FRESH_DIR / ("primary_rep%d.jsonl" % r)
                for r in REP_MAP["fresh"]["primary"]],
    "stress": [FRESH_DIR / ("stress_rep%d.jsonl" % r)
               for r in REP_MAP["fresh"]["stress"]],
}

HEADER_LINE = (
    r"Dataset & prior & $n_{\rm ens}$ & $K=8$ $C_\star$/iid $K=128$ primary"
    r" & $K=8$ $C_\star$/iid $K=128$ stress & iid $K=8$/iid $K=128$ stress"
    r" & $K=8$ $C_\star$/pair stress \\")
COLUMNS = [
    ("k8_cstar_over_iid_k128_primary", "$K=8$ $C_\\star$/iid $K=128$ primary"),
    ("k8_cstar_over_iid_k128_stress", "$K=8$ $C_\\star$/iid $K=128$ stress"),
    ("iid_k8_over_iid_k128_stress", "iid $K=8$/iid $K=128$ stress"),
    ("k8_cstar_over_pair_stress", "$K=8$ $C_\\star$/pair stress"),
]


class GateFailure(Exception):
    pass


def fail(msg):
    raise GateFailure(msg)


def prior_index(prior):
    if prior == 0.1:
        return 0
    if prior == 0.3:
        return 1
    fail("unexpected prior value %r" % (prior,))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_rel(path):
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def load_and_validate_file(path, block, expected_reps_in_block):
    """Fail-closed row validator (G1 spec + DR-4a/d).  Returns
    (replicate, {(dataset, prior_idx, method, chains): trace_var})."""
    path = Path(path)
    if not path.is_file():
        fail("input file missing: %s" % path)
    rows = []
    with open(path, "r") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                fail("%s: blank line at %d" % (path, i + 1))
            rows.append(json.loads(line))
    expected_set = PRIMARY_SET if block == "primary" else STRESS_SET
    expected_n = len(expected_set) * len(DATASET_ORDER) * len(PRIORS)
    if len(rows) != expected_n:
        fail("%s: expected %d rows for %s block, found %d"
             % (path, expected_n, block, len(rows)))
    reps_seen = set()
    tv = {}
    for i, row in enumerate(rows):
        where = "%s row %d" % (path.name, i + 1)
        if frozenset(row.keys()) != KEY_PIN:
            missing = sorted(KEY_PIN - set(row.keys()))
            extra = sorted(set(row.keys()) - KEY_PIN)
            fail("%s: key set != pinned 38-key schema (missing=%s extra=%s)"
                 % (where, missing, extra))
        for k, v in row.items():
            if isinstance(v, float) and not math.isfinite(v):
                fail("%s: nonfinite value in field %s" % (where, k))
        if row["target"] != "real_logistic":
            fail("%s: target %r != 'real_logistic'" % (where, row["target"]))
        if row["method"] != row["coupling"]:
            fail("%s: method %r != coupling %r"
                 % (where, row["method"], row["coupling"]))
        ds = row["dataset"]
        if ds not in DATASET_PIN:
            fail("%s: unknown dataset %r" % (where, ds))
        pidx = prior_index(row["prior"])
        label, reps8, dim, steps, burnin, ref_steps, ref_burnin = \
            DATASET_PIN[ds]
        chains = row["chains"]
        if chains not in (8, 128):
            fail("%s: chains %r not in {8, 128}" % (where, chains))
        want_reps = reps8 if chains == 8 else K128_REPS
        if row["reps"] != want_reps or row["n_ensembles"] != want_reps:
            fail("%s: reps/n_ensembles (%r, %r) != %d for %s@K=%d"
                 % (where, row["reps"], row["n_ensembles"], want_reps, ds,
                    chains))
        if row["eta"] != ETA_BY_PRIOR[pidx]:
            fail("%s: eta %r != %r for prior %r"
                 % (where, row["eta"], ETA_BY_PRIOR[pidx], row["prior"]))
        if (row["steps"], row["burnin"]) != (steps, burnin):
            fail("%s: steps/burnin (%r, %r) != (%d, %d)"
                 % (where, row["steps"], row["burnin"], steps, burnin))
        if (row["ref_steps"], row["ref_burnin"]) != (ref_steps, ref_burnin):
            fail("%s: ref_steps/ref_burnin (%r, %r) != (%d, %d)"
                 % (where, row["ref_steps"], row["ref_burnin"], ref_steps,
                    ref_burnin))
        if row["ref_reps"] != 16 or row["ref_chains"] != 16:
            fail("%s: ref_reps/ref_chains (%r, %r) != (16, 16)"
                 % (where, row["ref_reps"], row["ref_chains"]))
        if row["windows"] != row["steps"] - row["burnin"]:
            fail("%s: windows %r != steps - burnin = %d"
                 % (where, row["windows"], row["steps"] - row["burnin"]))
        if row["dim"] != dim:
            fail("%s: dim %r != %d" % (where, row["dim"], dim))
        rep = row["replicate"]
        reps_seen.add(rep)
        # DR-4d: exact seed/design_seed recomputation (holds for the
        # archived rows too -- verified 126/126 -- so enforced in both modes)
        want_seed = SEED_BASE[ds] + 10000 * rep + pidx
        want_design = SEED_BASE[ds] + pidx
        if row["seed"] != want_seed:
            fail("%s: seed %r != recomputed %d (base %d + 10000*%d + %d)"
                 % (where, row["seed"], want_seed, SEED_BASE[ds], rep, pidx))
        if row["design_seed"] != want_design:
            fail("%s: design_seed %r != recomputed %d"
                 % (where, row["design_seed"], want_design))
        if not (isinstance(row["trace_var"], float) and row["trace_var"] > 0):
            fail("%s: trace_var %r not a positive finite float"
                 % (where, row["trace_var"]))
        key = (ds, pidx, row["method"], chains)
        if key in tv:
            fail("%s: duplicate row for %s" % (where, (ds, row["prior"],
                                                       row["method"], chains)))
        tv[key] = row["trace_var"]
    if len(reps_seen) != 1:
        fail("%s: rows carry mixed replicate values %s"
             % (path, sorted(reps_seen)))
    rep = reps_seen.pop()
    if rep not in expected_reps_in_block:
        fail("%s: replicate %d not in registered %s replicate set %s "
             "(wrong replicate set)"
             % (path, rep, block, list(expected_reps_in_block)))
    for ds in DATASET_ORDER:
        for prior, _ in PRIORS:
            pidx = prior_index(prior)
            have = set((m, c) for (d, p, m, c) in tv
                       if d == ds and p == pidx)
            if have != set(expected_set):
                fail("%s: cell (%s, %s) method/chain set %s != expected %s "
                     "(missing rows)"
                     % (path, ds, prior, sorted(have), sorted(expected_set)))
    return rep, tv


def load_block(paths, block, mode):
    """Validate the 3 files of a block; enforce the registered replicate
    map for the mode.  Returns {replicate: trace_var dict}."""
    registered = REP_MAP[mode][block]
    if len(paths) != 3:
        fail("%s block needs exactly 3 files, got %d" % (block, len(paths)))
    by_rep = {}
    for p in paths:
        rep, tv = load_and_validate_file(p, block, set(registered))
        if rep in by_rep:
            fail("duplicate replicate %d in %s block" % (rep, block))
        by_rep[rep] = tv
    if set(by_rep) != set(registered):
        fail("%s block replicate set %s != registered %s map %s "
             "(wrong replicate set)"
             % (block, sorted(by_rep), mode, list(registered)))
    return by_rep


def med_min_max(vals):
    if len(vals) != 3:
        fail("expected 3 per-replicate values, got %d" % len(vals))
    s = sorted(vals)
    return s[1], s[0], s[2]


def compute_cells(primary_by_rep, stress_by_rep, mode):
    """Per-cell per-replicate ratios (same-replicate matching; trace_var
    exactly as emitted) -> {(ds, pidx, col_id): (med, mn, mx, {rep: val})}."""
    p_reps = sorted(REP_MAP[mode]["primary"])
    s_reps = sorted(REP_MAP[mode]["stress"])
    cells = {}
    for ds in DATASET_ORDER:
        for prior, _ in PRIORS:
            pidx = prior_index(prior)

            def ratios(reps, by_rep, num_key, den_key):
                out = {}
                for r in reps:
                    tv = by_rep[r]
                    out[r] = (tv[(ds, pidx) + num_key]
                              / tv[(ds, pidx) + den_key])
                return out

            # primary col: zero_sum@8 / iid@128 over PRIMARY replicates
            per = ratios(p_reps, primary_by_rep, ("zero_sum", 8), ("iid", 128))
            cells[(ds, pidx, "k8_cstar_over_iid_k128_primary")] = \
                med_min_max(list(per.values())) + (per,)
            # stress col: zero_sum@8 / iid@128 over STRESS replicates
            per = ratios(s_reps, stress_by_rep, ("zero_sum", 8), ("iid", 128))
            cells[(ds, pidx, "k8_cstar_over_iid_k128_stress")] = \
                med_min_max(list(per.values())) + (per,)
            # iid K=8 / iid K=128: STRESS replicates only
            per = ratios(s_reps, stress_by_rep, ("iid", 8), ("iid", 128))
            cells[(ds, pidx, "iid_k8_over_iid_k128_stress")] = \
                med_min_max(list(per.values())) + (per,)
            # K=8 C*/pair: zero_sum@8 / antithetic@8, STRESS ONLY (the
            # primary antithetic rows exist but MUST NOT be used -- G1 spec)
            per = ratios(s_reps, stress_by_rep, ("zero_sum", 8),
                         ("antithetic", 8))
            cells[(ds, pidx, "k8_cstar_over_pair_stress")] = \
                med_min_max(list(per.values())) + (per,)
    return cells


def n_ens_string(ds):
    return "%d/%d" % (DATASET_PIN[ds][1], K128_REPS)


def fmt_cell(med, mn, mx):
    return "%s [%s, %s]" % (format(med, ".4f"), format(mn, ".4f"),
                            format(mx, ".4f"))


def render(cells):
    lines = [r"\begin{tabular}{llcllll}", r"\toprule", HEADER_LINE,
             r"\midrule"]
    for ds in DATASET_ORDER:
        label = DATASET_PIN[ds][0]
        for prior, prior_str in PRIORS:
            pidx = prior_index(prior)
            parts = [label, prior_str, n_ens_string(ds)]
            for col_id, _ in COLUMNS:
                med, mn, mx, _ = cells[(ds, pidx, col_id)]
                parts.append(fmt_cell(med, mn, mx))
            lines.append(" & ".join(parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines).encode("ascii")


CELL_RE = re.compile(r"^(-?\d+\.\d{4}) \[(-?\d+\.\d{4}), (-?\d+\.\d{4})\]$")


def structural_self_check(tex_bytes, cells, tag):
    """DR-4 self-checks: 6 data rows; n_ens strings; re-parse emitted
    floats == computed.  Returns printable report lines."""
    body = tex_bytes.decode("ascii").split("\n")
    i_mid, i_bot = body.index(r"\midrule"), body.index(r"\bottomrule")
    data = body[i_mid + 1:i_bot]
    if len(data) != 6:
        fail("self-check[%s]: %d data rows != 6" % (tag, len(data)))
    nens_seen = []
    expect_rows = [(ds, prior_str) for ds in DATASET_ORDER
                   for _, prior_str in PRIORS]
    for line, (ds, prior_str) in zip(data, expect_rows):
        if not line.endswith(r" \\"):
            fail("self-check[%s]: row missing trailing ' \\\\': %r"
                 % (tag, line))
        parts = line[:-3].split(" & ")
        if len(parts) != 7:
            fail("self-check[%s]: row has %d fields != 7" % (tag, len(parts)))
        if parts[0] != DATASET_PIN[ds][0] or parts[1] != prior_str:
            fail("self-check[%s]: row order mismatch: %r" % (tag, parts[:2]))
        if parts[2] != n_ens_string(ds):
            fail("self-check[%s]: n_ens %r != %r"
                 % (tag, parts[2], n_ens_string(ds)))
        nens_seen.append(parts[2])
        pidx = prior_index(0.1 if prior_str == "0.1" else 0.3)
        for cell_text, (col_id, _) in zip(parts[3:], COLUMNS):
            m = CELL_RE.match(cell_text)
            if not m:
                fail("self-check[%s]: unparseable cell %r" % (tag, cell_text))
            med, mn, mx, _ = cells[(ds, pidx, col_id)]
            want = tuple(format(v, ".4f") for v in (med, mn, mx))
            if m.groups() != want:
                fail("self-check[%s]: re-parsed cell %s != computed %s at "
                     "(%s, %s, %s)" % (tag, m.groups(), want, ds, prior_str,
                                       col_id))
    return [
        "SELF-CHECK[%s]: 6 data rows OK" % tag,
        "SELF-CHECK[%s]: n_ens strings %s OK" % (tag, nens_seen),
        "SELF-CHECK[%s]: re-parsed emitted floats == computed (24 ranged "
        "cells, 72 numbers) OK" % tag,
    ]


def run_archived(primary_paths, stress_paths):
    """Archived gate (DR-4b).  Returns (cells, tex_bytes, check_lines)."""
    primary = load_block(primary_paths, "primary", "archived")
    stress = load_block(stress_paths, "stress", "archived")
    cells = compute_cells(primary, stress, "archived")
    tex = render(cells)
    frozen = FROZEN_TEX.read_bytes()
    if tex != frozen:
        n = min(len(tex), len(frozen))
        at = next((i for i in range(n) if tex[i] != frozen[i]), n)
        fail("ARCHIVED GATE FAIL: emitted bytes != %s (first difference at "
             "byte %d; emitted %d bytes, frozen %d bytes)"
             % (FROZEN_TEX, at, len(tex), len(frozen)))
    checks = structural_self_check(tex, cells, "archived")
    return cells, tex, checks


def paper_quoted_scalar_names():
    # The two paper-quoted (Results-paragraph) scalars: WDBC/0.3 primary + stress medians
    # (printed 0.1189 / 0.1344) -- DR-7c.
    return [("wdbc", 1, "k8_cstar_over_iid_k128_primary", "0.1189"),
            ("wdbc", 1, "k8_cstar_over_iid_k128_stress", "0.1344")]


def build_comparison(frozen_cells, fresh_cells, fresh_paths):
    cols = dict(COLUMNS)
    cell_records = []
    confirm = disagree = 0
    for ds in DATASET_ORDER:
        for prior, prior_str in PRIORS:
            pidx = prior_index(prior)
            for col_id, _ in COLUMNS:
                fz_med, fz_mn, fz_mx, _ = frozen_cells[(ds, pidx, col_id)]
                fr_med, fr_mn, fr_mx, fr_per = fresh_cells[(ds, pidx, col_id)]
                ok = fr_mn <= fz_med <= fr_mx
                confirm, disagree = confirm + ok, disagree + (not ok)
                cell_records.append({
                    "dataset": DATASET_PIN[ds][0],
                    "prior": prior_str,
                    "column": col_id,
                    "column_header": cols[col_id],
                    "frozen_printed": fmt_cell(fz_med, fz_mn, fz_mx),
                    "frozen_full_precision": {"median": fz_med, "min": fz_mn,
                                              "max": fz_mx},
                    "fresh": {"median": fr_med, "min": fr_mn, "max": fr_mx,
                              "per_replicate": {str(r): v for r, v
                                                in sorted(fr_per.items())}},
                    "verdict": "confirm" if ok else "disagree",
                })
    nens_records = [{"dataset": DATASET_PIN[ds][0], "prior": prior_str,
                     "frozen": n_ens_string(ds), "fresh": n_ens_string(ds),
                     "verdict": "equal"}
                    for ds in DATASET_ORDER for _, prior_str in PRIORS]
    scalar_records = []
    for ds, pidx, col_id, printed in paper_quoted_scalar_names():
        fz_med = frozen_cells[(ds, pidx, col_id)][0]
        fr_med, fr_mn, fr_mx, _ = fresh_cells[(ds, pidx, col_id)]
        rel = abs(fr_med - fz_med) / abs(fz_med)
        scalar_records.append({
            "name": "%s prior 0.3 %s median (paper-quoted, Results paragraph main.tex ~line 593; not the abstract at current HEAD)"
                    % (DATASET_PIN[ds][0], col_id),
            "frozen_printed_median": printed,
            "frozen_full_precision_median": fz_med,
            "fresh_median": fr_med, "fresh_min": fr_mn, "fresh_max": fr_mx,
            "clause_i_frozen_median_within_fresh_minmax":
                bool(fr_mn <= fz_med <= fr_mx),
            "clause_ii_relative_diff": rel,
            "clause_ii_within_15pct": bool(rel <= 0.15),
        })
    record = {
        "schema": "camera-ready-regen-comparison-highk-v1",
        "table": "campaign_highk_iid",
        "frozen_tex": str(FROZEN_TEX.relative_to(REPO)),
        "frozen_tex_sha256": sha256_file(FROZEN_TEX),
        "archived_gate": "PASS (byte-identical)",
        "match_rule": {
            "ranged_cells": "confirm iff frozen full-precision median (DR-7a,"
                " recomputed from archived raws by this gated generator) lies"
                " within [fresh_min, fresh_max] of the fresh per-replicate"
                " ratios (precommit protocol item 5; AM-13)",
            "paper_quoted_scalars": "the two WDBC/0.3 medians quoted in the paper's"
                " Results paragraph (main.tex ~line 593; NOT the abstract at current"
                " HEAD) get BOTH clauses reported (DR-7c): (i)"
                " frozen median within fresh min-max; (ii)"
                " |fresh_med - frozen| / |frozen| <= 0.15 (DR-7b)",
            "layer_framing": "AM-0: archived-mode byte gate = RESTORATION layer;"
                " this record = CONFIRMATION layer",
            "vr2_seed_sharing": "intentional same-design sharing (shared +77717"
                " reference stream across methods within a cell; AM-10(i) K=8/K=128"
                " base-seed sharing at the same cell) vs accidental cross-target"
                " adjacent-dataset/adjacent-replicate coincidences (DR-3) — both"
                " reproduce the June structure; envelopes disjoint (VR-2)",
            "n_ens_cells": "categorical equality, descriptive",
            "pass_fraction_register": "descriptive only, never calibrated"
                " statistics (DR-8)",
        },
        "replicates": {
            "archived_primary": list(REP_MAP["archived"]["primary"]),
            "archived_stress": list(REP_MAP["archived"]["stress"]),
            "fresh_primary": list(REP_MAP["fresh"]["primary"]),
            "fresh_stress": list(REP_MAP["fresh"]["stress"]),
        },
        "inputs": {
            "archived": {repo_rel(p): sha256_file(p)
                         for b in ("primary", "stress")
                         for p in ARCHIVED_FILES[b]},
            "fresh": {repo_rel(p): sha256_file(p)
                      for b in ("primary", "stress") for p in fresh_paths[b]},
        },
        "cells": cell_records,
        "n_ens_cells": nens_records,
        "paper_quoted_scalars": scalar_records,
        "summary": {"ranged_cells": 24, "confirm": confirm,
                    "disagree": disagree},
    }
    return record


def run_fresh(primary_paths, stress_paths):
    # 1. Fresh raws: full validator battery under the fresh registered map.
    primary = load_block(primary_paths, "primary", "fresh")
    stress = load_block(stress_paths, "stress", "fresh")
    # 2. Full-precision frozen comparanda via the in-memory archived gate
    #    (DR-7a: "via the gated generators"); any archived failure aborts.
    frozen_cells, _, _ = run_archived(ARCHIVED_FILES["primary"],
                                      ARCHIVED_FILES["stress"])
    # 3. Fresh table + self-checks.
    fresh_cells = compute_cells(primary, stress, "fresh")
    fresh_tex = render(fresh_cells)
    checks = structural_self_check(fresh_tex, fresh_cells, "fresh")
    # 4. Comparison record (protocol item 5 + AM-13 + DR-7).
    record = build_comparison(frozen_cells, fresh_cells,
                              {"primary": primary_paths,
                               "stress": stress_paths})
    # 5. Emit ONLY under outputs/sweeps/camera_ready_regen/highk/tables/
    #    (DR-4e), after every check has passed.
    FRESH_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = FRESH_TABLES_DIR / "campaign_highk_iid_fresh.tex"
    json_path = FRESH_TABLES_DIR / "comparison_highk.json"
    tex_path.write_bytes(fresh_tex)
    json_path.write_text(json.dumps(record, indent=2) + "\n")
    print("FRESH MODE COMPLETE")
    for line in checks:
        print(line)
    print("wrote %s" % tex_path)
    print("wrote %s" % json_path)
    print("comparison summary: %d/%d ranged cells confirm (descriptive; "
          "DR-8)" % (record["summary"]["confirm"],
                     record["summary"]["ranged_cells"]))
    for s in record["paper_quoted_scalars"]:
        print("paper-quoted scalar %s: clause_i=%s clause_ii=%s "
              "(rel diff %.6f)"
              % (s["frozen_printed_median"],
                 s["clause_i_frozen_median_within_fresh_minmax"],
                 s["clause_ii_within_15pct"], s["clause_ii_relative_diff"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", required=True, choices=["archived", "fresh"])
    ap.add_argument("--primary", nargs=3, metavar="JSONL",
                    help="override the 3 primary replicate files")
    ap.add_argument("--stress", nargs=3, metavar="JSONL",
                    help="override the 3 stress replicate files")
    ap.add_argument("--emit", metavar="PATH",
                    help="archived mode only: also write the gated bytes "
                         "here (must NOT be under paper/)")
    args = ap.parse_args()
    try:
        defaults = ARCHIVED_FILES if args.mode == "archived" else FRESH_FILES
        primary = [Path(p) for p in args.primary] if args.primary \
            else defaults["primary"]
        stress = [Path(p) for p in args.stress] if args.stress \
            else defaults["stress"]
        if args.mode == "archived":
            cells, tex, checks = run_archived(primary, stress)
            print("ARCHIVED GATE PASS (byte-identical)")
            for line in checks:
                print(line)
            for _, pidx, col_id, printed in paper_quoted_scalar_names():
                print("EXPORT WDBC/0.3 %s median (printed %s) full "
                      "precision: %.17g"
                      % (col_id, printed, cells[("wdbc", pidx, col_id)][0]))
            if args.emit:
                out = Path(args.emit).resolve()
                paper_dir = (REPO / "paper").resolve()
                if paper_dir == out or paper_dir in out.parents:
                    fail("--emit refuses to write under paper/: %s" % out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tex)
                print("emitted archived-mode bytes to %s" % out)
        else:
            if args.emit:
                fail("--emit is an archived-mode flag; fresh outputs are "
                     "pinned to %s (DR-4e)" % FRESH_TABLES_DIR)
            run_fresh(primary, stress)
    except GateFailure as e:
        print("FAIL: %s" % e)
        sys.exit(1)


if __name__ == "__main__":
    main()
