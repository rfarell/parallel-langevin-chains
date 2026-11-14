#!/usr/bin/env python3
"""Registered G3 generator for paper/tables/campaign_wallclock_frontier.tex.

Clone of campaigns/analyze_attempt08_wallclock_frontier.py (the surviving
June analyzer) per campaigns/camera_ready_regen_invocation_plan.md
SECTION 2 "### G3" (Revision 5) and campaigns/camera_ready_regen_precommit.md
rulings DR-4, DR-5, DR-6, DR-7(d,e), VR-4, VR-5, VR-7.  Deterministic, CPU-only.

Both modes are fail-closed: any check failure => NO output file written,
nonzero exit.

--mode archived
    Inputs: campaigns/attempt08/{primary,stress,rand,scale1,k5}_rep*.jsonl
    (replicates 300-329, the analyzer's exact block ranges).  Rebuilds the
    analyzer result markdown AND a NEW worst-cell .tex (the June emitter was a
    one-off hand transcription; this emitter is templated from the frozen
    file), both gated BYTE-IDENTICAL against the frozen committed artifacts
    (DR-4b).  Emits only under --out-dir; never paper/tables/ or
    campaigns/.

--mode fresh
    Inputs: outputs/sweeps/camera_ready_regen/frontier/ replicates
    primary 40110-40115, stress 40116-40121, rand 40122-40127,
    scale1 40128-40133, k5 40134-40139 (+ optional smoke 40109, validated but
    excluded from scoring).  Enforces the VR-4 all-block exact-hostname rule,
    the registered block->GPU uuid map via launch sidecars
    (VR-1(f)/VR-5), and DR-4(d) exact per-row seed / design_seed / order_seed
    recomputation.  Re-runs the archived byte gates in memory first (DR-4b:
    gate before being trusted on fresh raws), then emits
    outputs/sweeps/camera_ready_regen/frontier/tables/
    campaign_wallclock_frontier_fresh.tex (label register "same-host
    cross-device stress" per DR-5) and comparison_frontier.json (DR-7(d,e):
    exactly 14 numeric outcomes).

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

import numpy as np


REPO = Path(__file__).resolve().parents[1]
HARNESS_PATH = REPO / "campaigns" / "real_logistic_wallclock_gpu.py"
HARNESS_BASENAME = "real_logistic_wallclock_gpu.py"
FROZEN_TEX = REPO / "paper" / "tables" / "campaign_wallclock_frontier.tex"
FROZEN_MD = REPO / "campaigns" / "attempt08_wallclock_frontier_result.md"
DEFAULT_ARCHIVED_DIR = REPO / "campaigns" / "attempt08"
DEFAULT_FRESH_DIR = REPO / "outputs" / "sweeps" / "camera_ready_regen" / "frontier"
FRESH_TABLES_DIR = REPO / "outputs" / "sweeps" / "camera_ready_regen" / "frontier" / "tables"
EXPECTED_SCRIPT_HASH = "00709cfe7538e7c9"
FRESH_HOST_SHORT = "redacted-hpc-node"

# ---------------------------------------------------------------------------
# Verbatim clone of the June analyzer (campaigns/
# analyze_attempt08_wallclock_frontier.py); BLOCKS parameterized per G3(a,b).
# ---------------------------------------------------------------------------

DATASETS = ["wdbc", "spambase", "adult"]
PRIORS = [0.1, 0.3]
FULL_IID_K = [8, 16, 32, 64, 128]
FULL_CONFIGS = [
    (7, "iid"),
    (7, "zero_sum"),
    (7, "antithetic"),
    (8, "iid"),
    (8, "zero_sum"),
    (8, "antithetic"),
    (16, "iid"),
    (32, "iid"),
    (64, "iid"),
    (128, "iid"),
]
K5_CONFIGS = [(5, "iid"), (5, "zero_sum"), (5, "antithetic")]
ARCHIVED_BLOCKS = {
    "primary": {"scale": 0.0, "mode": "full", "reps": list(range(300, 306)), "host_prefix": "redacted-hpc-node", "floor": 0.10},
    "stress": {"scale": 0.0, "mode": "full", "reps": list(range(306, 312)), "host_prefix": "redacted-hpc-node", "floor": 0.10},
    "rand": {"scale": 0.25, "mode": "full", "reps": list(range(312, 318)), "host_prefix": None, "floor": 0.10},
    "scale1": {"scale": 1.0, "mode": "full", "reps": list(range(318, 324)), "host_prefix": None, "floor": 0.10},
    "k5": {"scale": 0.0, "mode": "k5", "reps": list(range(324, 330)), "host_prefix": None, "floor": 0.10},
}
# Fresh replicate map per AM-1(c); host_prefix None because fresh mode replaces
# the archived prefix rule with the VR-4 all-block exact-hostname check.
FRESH_BLOCKS = {
    "primary": {"scale": 0.0, "mode": "full", "reps": list(range(40110, 40116)), "host_prefix": None, "floor": 0.10},
    "stress": {"scale": 0.0, "mode": "full", "reps": list(range(40116, 40122)), "host_prefix": None, "floor": 0.10},
    "rand": {"scale": 0.25, "mode": "full", "reps": list(range(40122, 40128)), "host_prefix": None, "floor": 0.10},
    "scale1": {"scale": 1.0, "mode": "full", "reps": list(range(40128, 40134)), "host_prefix": None, "floor": 0.10},
    "k5": {"scale": 0.0, "mode": "k5", "reps": list(range(40134, 40140)), "host_prefix": None, "floor": 0.10},
}
EXPECTED_HORIZON = {
    "wdbc": {"steps": 900, "burnin": 450, "ref_steps": 1800, "ref_burnin": 900, "base_reps": 256},
    "spambase": {"steps": 900, "burnin": 450, "ref_steps": 1800, "ref_burnin": 900, "base_reps": 192},
    "adult": {"steps": 700, "burnin": 350, "ref_steps": 1400, "ref_burnin": 700, "base_reps": 64},
}


def expected_reps(dataset, chains):
    base = EXPECTED_HORIZON[dataset]["base_reps"]
    return max(24, base // max(1, chains // 8))


def load_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def median(values):
    return float(np.median(np.asarray(values, dtype=float)))


def fmt(value):
    if value is None or not np.isfinite(value):
        return "--"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.4f}"


def finite_row(row):
    for value in row.values():
        if isinstance(value, (int, float)) and not np.isfinite(float(value)):
            return False
    return True


def cost(row):
    return float(row["trace_var"]) * float(row["elapsed_sec_per_ensemble"])


def row_key(row):
    return (
        row["dataset"],
        float(row["prior"]),
        int(row["chains"]),
        row["method"],
        int(row["replicate"]),
        float(row["init_scale"]),
        row["mode"],
    )


def metadata_present(row):
    fields = [
        "hostname",
        "gpu_name",
        "gpu_uuid",
        "driver_version",
        "cupy_version",
        "cuda_runtime_version",
        "order_seed",
        "config_index",
        "order_index",
        "elapsed_sec",
        "elapsed_sec_per_ensemble",
    ]
    for field in fields:
        if field not in row:
            return False
        value = row[field]
        if isinstance(value, str) and not value:
            return False
    return float(row["elapsed_sec"]) > 0.0 and float(row["elapsed_sec_per_ensemble"]) > 0.0


def validate_rows(rows):
    seen = {}
    duplicates = []
    nonfinite = []
    protocol_errors = []
    script_hashes = set()
    for row in rows:
        key = row_key(row)
        if key in seen:
            duplicates.append(key)
        seen[key] = row
        if not finite_row(row):
            nonfinite.append(key)
        script_hashes.add(row.get("script_hash"))
        dataset = row.get("dataset")
        expected = EXPECTED_HORIZON.get(dataset, {})
        chains = int(row.get("chains", -1))
        mode = row.get("mode")
        method = row.get("method")
        checks = {
            "target": row.get("target") == "real_logistic_wallclock_ula",
            "method": method in {"iid", "zero_sum", "antithetic"},
            "coupling": row.get("coupling") == method,
            "mode": mode in {"full", "k5"},
            "chains": (mode == "full" and chains in {7, 8, 16, 32, 64, 128}) or (mode == "k5" and chains == 5),
            "method_by_chains": (method == "iid" or chains in {5, 7, 8}),
            "init_scale": float(row.get("init_scale")) in {0.0, 0.25, 1.0},
            "steps": int(row.get("steps")) == expected.get("steps"),
            "burnin": int(row.get("burnin")) == expected.get("burnin"),
            "ref_steps": int(row.get("ref_steps")) == expected.get("ref_steps"),
            "ref_burnin": int(row.get("ref_burnin")) == expected.get("ref_burnin"),
            "ref_reps": int(row.get("ref_reps")) == 16,
            "ref_chains": int(row.get("ref_chains")) == 16,
            "n_ensembles": int(row.get("n_ensembles")) == expected_reps(dataset, chains),
            "windows": int(row.get("windows")) == expected.get("steps") - expected.get("burnin"),
            "script_hash": isinstance(row.get("script_hash"), str) and len(row.get("script_hash")) == 16,
            "metadata": metadata_present(row),
        }
        failed = [field for field, ok in checks.items() if not ok]
        if failed:
            protocol_errors.append((key, failed))
    return seen, duplicates, nonfinite, protocol_errors, script_hashes


def get_row(rows_by_key, dataset, prior, chains, method, rep, scale, mode):
    return rows_by_key.get((dataset, float(prior), int(chains), method, int(rep), float(scale), mode))


def expected_configs(mode):
    if mode == "full":
        return FULL_CONFIGS
    if mode == "k5":
        return K5_CONFIGS
    raise ValueError(mode)


def required_missing(rows_by_key, block, spec):
    missing = []
    host_errors = []
    for dataset in DATASETS:
        for prior in PRIORS:
            for rep in spec["reps"]:
                for chains, method in expected_configs(spec["mode"]):
                    row = get_row(rows_by_key, dataset, prior, chains, method, rep, spec["scale"], spec["mode"])
                    if row is None:
                        missing.append((block, dataset, prior, chains, method, rep, spec["scale"], spec["mode"]))
                    elif spec["host_prefix"] and not str(row.get("hostname", "")).startswith(spec["host_prefix"]):
                        host_errors.append((block, row.get("hostname"), dataset, prior, chains, method, rep))
    return missing, host_errors


def full_block_cells(rows_by_key, block, spec):
    cells = []
    for dataset in DATASETS:
        for prior in PRIORS:
            high = []
            odd = []
            even = []
            iid_trace_scaling = []
            selected_speedups = []
            selected_best_k = []
            single = []
            marginal2 = []
            logloss = []
            mse = []
            for rep in spec["reps"]:
                z8 = get_row(rows_by_key, dataset, prior, 8, "zero_sum", rep, spec["scale"], "full")
                a8 = get_row(rows_by_key, dataset, prior, 8, "antithetic", rep, spec["scale"], "full")
                iid8 = get_row(rows_by_key, dataset, prior, 8, "iid", rep, spec["scale"], "full")
                z7 = get_row(rows_by_key, dataset, prior, 7, "zero_sum", rep, spec["scale"], "full")
                a7 = get_row(rows_by_key, dataset, prior, 7, "antithetic", rep, spec["scale"], "full")
                iid7 = get_row(rows_by_key, dataset, prior, 7, "iid", rep, spec["scale"], "full")
                iid_frontier = [
                    get_row(rows_by_key, dataset, prior, chains, "iid", rep, spec["scale"], "full")
                    for chains in FULL_IID_K
                ]
                iid_costs = [(chains, row, cost(row)) for chains, row in zip(FULL_IID_K, iid_frontier)]
                best_k, best_iid, best_cost = min(iid_costs, key=lambda item: item[2])
                high.append(cost(z8) / best_cost)
                odd.append(cost(z7) / min(cost(iid7), cost(a7)))
                even.append(cost(z8) / cost(a8))
                iid128 = get_row(rows_by_key, dataset, prior, 128, "iid", rep, spec["scale"], "full")
                iid_trace_scaling.append(float(iid8["trace_var"]) / float(iid128["trace_var"]))
                selected_speedups.append(float(best_iid["elapsed_sec_per_ensemble"]) / float(z8["elapsed_sec_per_ensemble"]))
                selected_best_k.append(best_k)
                for chains, zrow, iidrow in [(7, z7, iid7), (8, z8, iid8)]:
                    single.append(float(zrow["single_trace_var"]) / float(iidrow["single_trace_var"]))
                    marginal2.append(float(zrow["marginal_second"]) / float(iidrow["marginal_second"]))
                    logloss.append(float(zrow["test_logloss"]) / float(iidrow["test_logloss"]))
                    mse.append(float(zrow["mse"]) / float(iidrow["mse"]))
            high_med = median(high)
            odd_med = median(odd)
            cells.append(
                {
                    "block": block,
                    "dataset": dataset,
                    "prior": prior,
                    "reps": spec["reps"],
                    "scale": spec["scale"],
                    "mode": "full",
                    "score": max(high_med, odd_med),
                    "high": high_med,
                    "odd": odd_med,
                    "even_zero_ant": median(even),
                    "iid_trace_scaling": median(iid_trace_scaling),
                    "selected_speedup": median(selected_speedups),
                    "selected_best_k_median": median(selected_best_k),
                    "single_min": min(single),
                    "single_max": max(single),
                    "marginal2_min": min(marginal2),
                    "marginal2_max": max(marginal2),
                    "logloss_max": max(logloss),
                    "mse_max": max(mse),
                }
            )
    return cells


def k5_block_cells(rows_by_key, block, spec):
    cells = []
    for dataset in DATASETS:
        for prior in PRIORS:
            ratios = []
            even = []
            single = []
            marginal2 = []
            logloss = []
            mse = []
            for rep in spec["reps"]:
                z5 = get_row(rows_by_key, dataset, prior, 5, "zero_sum", rep, spec["scale"], "k5")
                a5 = get_row(rows_by_key, dataset, prior, 5, "antithetic", rep, spec["scale"], "k5")
                iid5 = get_row(rows_by_key, dataset, prior, 5, "iid", rep, spec["scale"], "k5")
                ratios.append(cost(z5) / min(cost(iid5), cost(a5)))
                even.append(cost(z5) / cost(a5))
                single.append(float(z5["single_trace_var"]) / float(iid5["single_trace_var"]))
                marginal2.append(float(z5["marginal_second"]) / float(iid5["marginal_second"]))
                logloss.append(float(z5["test_logloss"]) / float(iid5["test_logloss"]))
                mse.append(float(z5["mse"]) / float(iid5["mse"]))
            cells.append(
                {
                    "block": block,
                    "dataset": dataset,
                    "prior": prior,
                    "reps": spec["reps"],
                    "scale": spec["scale"],
                    "mode": "k5",
                    "score": median(ratios),
                    "high": math.nan,
                    "odd": median(ratios),
                    "even_zero_ant": median(even),
                    "iid_trace_scaling": math.nan,
                    "selected_speedup": math.nan,
                    "selected_best_k_median": math.nan,
                    "single_min": min(single),
                    "single_max": max(single),
                    "marginal2_min": min(marginal2),
                    "marginal2_max": max(marginal2),
                    "logloss_max": max(logloss),
                    "mse_max": max(mse),
                }
            )
    return cells


def worst(cells, key):
    values = [cell[key] for cell in cells]
    if not values or any(not np.isfinite(value) for value in values):
        return float("inf")
    return max(values)


def min_value(cells, key):
    values = [cell[key] for cell in cells if np.isfinite(cell[key])]
    if not values:
        return float("-inf")
    return min(values)


def render_cells(lines, title, cells):
    lines.extend(
        [
            f"## {title}",
            "",
            "| block | mode | scale | dataset | prior | reps | score | high z8/best iid cost | odd z/best odd cost | z8/ant8 cost | iid8/iid128 trace | best-iid speedup | best-iid K med | single range | marginal2 range | logloss max | mse max |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        lines.append(
            "| {block} | {mode} | {scale:.2f} | {dataset} | {prior:.1f} | {reps} | {score} | {high} | {odd} | {even_zero_ant} | "
            "{iid_trace_scaling} | {selected_speedup} | {selected_best_k_median} | [{single_min}, {single_max}] | "
            "[{marginal2_min}, {marginal2_max}] | {logloss_max} | {mse_max} |".format(
                block=cell["block"],
                mode=cell["mode"],
                scale=cell["scale"],
                dataset=cell["dataset"],
                prior=cell["prior"],
                reps=",".join(str(rep) for rep in cell["reps"]),
                score=fmt(cell["score"]),
                high=fmt(cell["high"]),
                odd=fmt(cell["odd"]),
                even_zero_ant=fmt(cell["even_zero_ant"]),
                iid_trace_scaling=fmt(cell["iid_trace_scaling"]),
                selected_speedup=fmt(cell["selected_speedup"]),
                selected_best_k_median=fmt(cell["selected_best_k_median"]),
                single_min=fmt(cell["single_min"]),
                single_max=fmt(cell["single_max"]),
                marginal2_min=fmt(cell["marginal2_min"]),
                marginal2_max=fmt(cell["marginal2_max"]),
                logloss_max=fmt(cell["logloss_max"]),
                mse_max=fmt(cell["mse_max"]),
            )
        )
    lines.append("")


# Section titles / gate labels: archived strings are the June analyzer's exact
# strings (byte gate); fresh strings use the DR-5 register.
ARCHIVED_MD_LABELS = {"stress_gate": "cross-node stress F8", "stress_title": "Cross-Node Stress"}
FRESH_MD_LABELS = {"stress_gate": "same-host cross-device stress F8", "stress_title": "Same-Host Cross-Device Stress"}


def build_result_md(rows, cells_by_block, blocks, script_hashes, missing, duplicates, nonfinite, protocol_errors, host_errors, md_labels):
    """Verbatim clone of the June analyzer's report construction (main())."""
    scores = {name: worst(cells, "score") for name, cells in cells_by_block.items()}
    full_cells = cells_by_block["primary"] + cells_by_block["stress"] + cells_by_block["rand"] + cells_by_block["scale1"]
    all_cells = full_cells + cells_by_block["k5"]
    primary_ok = scores["primary"] <= blocks["primary"]["floor"]
    stress_ok = scores["stress"] <= blocks["stress"]["floor"]
    rand_ok = scores["rand"] <= blocks["rand"]["floor"]
    scale1_ok = scores["scale1"] <= blocks["scale1"]["floor"]
    k5_ok = scores["k5"] <= blocks["k5"]["floor"]
    even_low = min(min_value(cells_by_block[name], "even_zero_ant") for name in ["primary", "stress", "rand", "scale1"])
    even_high = max(worst(cells_by_block[name], "even_zero_ant") for name in ["primary", "stress", "rand", "scale1"])
    even_ok = np.isfinite(even_low) and even_low >= 0.5 and even_high <= 2.0
    iid_trace_scaling_min = min(min_value(cells_by_block[name], "iid_trace_scaling") for name in ["primary", "stress"])
    iid_trace_scaling_ok = iid_trace_scaling_min >= 4.0
    best_iid_speedup_min = min(min_value(cells_by_block[name], "selected_speedup") for name in ["primary", "stress"])
    best_iid_speedup_ok = best_iid_speedup_min >= 4.0
    single_ok = all(cell["single_min"] >= 0.7 and cell["single_max"] <= 1.3 for cell in full_cells)
    marginal_ok = all(cell["marginal2_min"] >= 0.7 and cell["marginal2_max"] <= 1.3 for cell in full_cells)
    logloss_ok = all(cell["logloss_max"] <= 1.10 for cell in full_cells)
    mse_ok = all(cell["mse_max"] <= 1.10 for cell in full_cells)
    finite_cells_ok = all(
        np.isfinite(value)
        for cell in all_cells
        for value in cell.values()
        if isinstance(value, (int, float)) and not math.isnan(value)
    )
    exact_ok = not missing and not duplicates and not nonfinite and not protocol_errors and not host_errors
    script_hash_ok = len(script_hashes) == 1 and all(isinstance(value, str) and len(value) == 16 for value in script_hashes)
    empirical_all_ok = all(
        [
            primary_ok,
            stress_ok,
            rand_ok,
            scale1_ok,
            k5_ok,
            even_ok,
            iid_trace_scaling_ok,
            best_iid_speedup_ok,
            single_ok,
            marginal_ok,
            logloss_ok,
            mse_ok,
            finite_cells_ok,
            exact_ok,
            script_hash_ok,
        ]
    )

    lines = [
        "# Attempt 8 Wall-Clock Frontier Result",
        "",
        "## Summary",
        "",
        f"- rows: {len(rows)}",
        f"- script_hashes: {', '.join(sorted(str(value) for value in script_hashes))}",
        f"- F8 primary: `{fmt(scores['primary'])}` (floor <= 0.10)",
        f"- F8_stress: `{fmt(scores['stress'])}` (floor <= 0.10)",
        f"- F8_rand: `{fmt(scores['rand'])}` (floor <= 0.10)",
        f"- F8_scale1: `{fmt(scores['scale1'])}` (floor <= 0.10)",
        f"- F8_k5: `{fmt(scores['k5'])}` (floor <= 0.10)",
        f"- even-K cost sanity range: `[{fmt(even_low)}, {fmt(even_high)}]` (required [0.5, 2.0])",
        f"- iid trace scaling min, primary/stress: `{fmt(iid_trace_scaling_min)}` (required >= 4.0)",
        f"- selected best-iid timing speedup min, primary/stress: `{fmt(best_iid_speedup_min)}` (required >= 4.0)",
        f"- exact/protocol ok: `{exact_ok}`; finite cells ok: `{finite_cells_ok}`; one script hash ok: `{script_hash_ok}`",
        f"- empirical gates all pass: `{empirical_all_ok}`",
        "",
        "## Gate Status",
        "",
        "| gate | pass | value | requirement |",
        "|---|---:|---:|---:|",
        f"| primary F8 | {primary_ok} | {fmt(scores['primary'])} | <= 0.10 |",
        f"| {md_labels['stress_gate']} | {stress_ok} | {fmt(scores['stress'])} | <= 0.10 |",
        f"| independent-start F8 | {rand_ok} | {fmt(scores['rand'])} | <= 0.10 |",
        f"| overdispersed-start F8 | {scale1_ok} | {fmt(scores['scale1'])} | <= 0.10 |",
        f"| K5 edge F8 | {k5_ok} | {fmt(scores['k5'])} | <= 0.10 |",
        f"| even-K cost sanity | {even_ok} | [{fmt(even_low)}, {fmt(even_high)}] | [0.5, 2.0] |",
        f"| iid trace scaling | {iid_trace_scaling_ok} | {fmt(iid_trace_scaling_min)} | >= 4.0 |",
        f"| selected best-iid timing speedup | {best_iid_speedup_ok} | {fmt(best_iid_speedup_min)} | >= 4.0 |",
        f"| single-chain guardrail | {single_ok} | -- | [0.7, 1.3] |",
        f"| marginal-second guardrail | {marginal_ok} | -- | [0.7, 1.3] |",
        f"| logloss guardrail | {logloss_ok} | -- | <= 1.10 |",
        f"| MSE guardrail | {mse_ok} | -- | <= 1.10 |",
        f"| exact/protocol | {exact_ok} | -- | no missing/duplicates/nonfinite/protocol/host errors |",
        f"| script hash | {script_hash_ok} | {len(script_hashes)} | exactly one valid hash |",
        "",
    ]
    render_cells(lines, "Primary", cells_by_block["primary"])
    render_cells(lines, md_labels["stress_title"], cells_by_block["stress"])
    render_cells(lines, "Independent-Start Stress", cells_by_block["rand"])
    render_cells(lines, "Overdispersed-Start Stress", cells_by_block["scale1"])
    render_cells(lines, "K5 Edge Stress", cells_by_block["k5"])

    if missing:
        lines.extend(["## Missing Rows", "", "```", *[repr(item) for item in missing[:200]], "```", ""])
    if host_errors:
        lines.extend(["## Host Errors", "", "```", *[repr(item) for item in host_errors[:200]], "```", ""])
    if duplicates:
        lines.extend(["## Duplicate Rows", "", "```", *[repr(item) for item in duplicates[:200]], "```", ""])
    if nonfinite:
        lines.extend(["## Nonfinite Rows", "", "```", *[repr(item) for item in nonfinite[:200]], "```", ""])
    if protocol_errors:
        lines.extend(["## Protocol Errors", "", "```", *[repr(item) for item in protocol_errors[:200]], "```", ""])

    summary = {
        "scores": {name: float(scores[name]) for name in scores},
        "floors": {name: float(blocks[name]["floor"]) for name in blocks},
        "block_floor_pass": {
            "primary": bool(primary_ok),
            "stress": bool(stress_ok),
            "rand": bool(rand_ok),
            "scale1": bool(scale1_ok),
            "k5": bool(k5_ok),
        },
        "even_k_cost_sanity": {"low": float(even_low), "high": float(even_high), "required": [0.5, 2.0], "pass": bool(even_ok)},
        "iid_trace_scaling_min": {"value": float(iid_trace_scaling_min), "required_min": 4.0, "pass": bool(iid_trace_scaling_ok)},
        "best_iid_speedup_min": {"value": float(best_iid_speedup_min), "required_min": 4.0, "pass": bool(best_iid_speedup_ok)},
        "single_chain_guardrail_pass": bool(single_ok),
        "marginal_second_guardrail_pass": bool(marginal_ok),
        "logloss_guardrail_pass": bool(logloss_ok),
        "mse_guardrail_pass": bool(mse_ok),
        "finite_cells_ok": bool(finite_cells_ok),
        "exact_protocol_ok": bool(exact_ok),
        "script_hash_ok": bool(script_hash_ok),
        "empirical_all_ok": bool(empirical_all_ok),
    }
    return "\n".join(lines) + "\n", summary


# ---------------------------------------------------------------------------
# NEW worst-cell .tex emitter (G3(c)): no June emitter survives; the frozen
# file was hand-transcribed, so the template below mirrors its exact bytes and
# is gated byte-identical against it in archived mode.
# ---------------------------------------------------------------------------

BLOCK_ORDER = ["primary", "stress", "rand", "scale1", "k5"]
ARCHIVED_TEX_LABELS = {
    "primary": "primary",
    "stress": "cross-node stress",
    "rand": "independent starts",
    "scale1": "overdispersed starts",
    "k5": "$K=5$ edge",
}
# DR-5: the fresh register replaces only the stress-block wording.
FRESH_TEX_LABELS = dict(ARCHIVED_TEX_LABELS, stress="same-host cross-device stress")
DATASET_DISPLAY = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}
TEX_HEADER_LINES = [
    "\\begin{tabular}{llclccc}",
    "\\toprule",
    "Block & Replicates & Start scale & Worst cell & $C_\\star K=8$/best iid cost & $C_\\star$ odd-$K$/best odd cost & Max variance-time ratio \\\\",
    "\\midrule",
]
TEX_FOOTER_LINES = ["\\bottomrule", "\\end{tabular}"]
# Archived-mode check values (G3 structural self-checks; must reproduce).
ARCHIVED_CHECK_VALUES = {
    "primary": ("0.0119", "0.0523", "K=7"),
    "stress": ("0.0114", "0.0555", "K=7"),
    "rand": ("0.0115", "0.0508", "K=7"),
    "scale1": ("0.0162", "0.0661", "K=7"),
    "k5": ("--", "0.0421", "K=5"),
}
ARCHIVED_WORST_CELL = "WDBC, 0.3"


def worst_cell_of(cells):
    # max() keeps the first argmax on exact ties, matching max(high, odd)
    # tie handling elsewhere; frozen data has a unique winner per block.
    return max(cells, key=lambda cell: cell["score"])


def cell_label(cell):
    return f"{DATASET_DISPLAY[cell['dataset']]}, {cell['prior']:.1f}"


def tex_row_fields(block, spec, cells, labels):
    cell = worst_cell_of(cells)
    score = worst(cells, "score")
    k_tag = "K=5" if spec["mode"] == "k5" else "K=7"
    high_txt = "--" if spec["mode"] == "k5" else f"{cell['high']:.4f}"
    odd_txt = f"{cell['odd']:.4f} (${k_tag}$)"
    return [
        labels[block],
        f"{spec['reps'][0]}--{spec['reps'][-1]}",
        f"{spec['scale']:.2f}",
        cell_label(cell),
        high_txt,
        odd_txt,
        f"{score:.4f}",
    ]


def build_tex(cells_by_block, blocks, labels):
    lines = list(TEX_HEADER_LINES)
    for block in BLOCK_ORDER:
        fields = tex_row_fields(block, blocks[block], cells_by_block[block], labels)
        lines.append(" & ".join(fields) + " \\\\")
    lines.extend(TEX_FOOTER_LINES)
    return "\n".join(lines) + "\n"


def reparse_tex(tex_text, cells_by_block, blocks, labels, errors):
    """Structural self-check: re-parse the emitted .tex against computed values."""
    lines = tex_text.splitlines()
    if lines[: len(TEX_HEADER_LINES)] != TEX_HEADER_LINES or lines[-len(TEX_FOOTER_LINES):] != TEX_FOOTER_LINES:
        errors.append("[TEX-SELFCHECK] header/footer lines do not match the template")
        return
    data_lines = lines[len(TEX_HEADER_LINES): -len(TEX_FOOTER_LINES)]
    if len(data_lines) != len(BLOCK_ORDER):
        errors.append(f"[TEX-SELFCHECK] expected {len(BLOCK_ORDER)} data rows, found {len(data_lines)}")
        return
    for block, line in zip(BLOCK_ORDER, data_lines):
        if not line.endswith(" \\\\"):
            errors.append(f"[TEX-SELFCHECK] row for {block} does not end with ' \\\\\\\\'")
            continue
        fields = line[: -len(" \\\\")].split(" & ")
        expected = tex_row_fields(block, blocks[block], cells_by_block[block], labels)
        if fields != expected:
            errors.append(f"[TEX-SELFCHECK] row for {block}: parsed {fields!r} != computed {expected!r}")


# ---------------------------------------------------------------------------
# Shared loading / validation scaffolding (fail-closed additions to the clone).
# ---------------------------------------------------------------------------

# Exact per-row key set, enumerated from the archived attempt08 raws (uniform
# across all archived rows including smoke); VR-3: FRONTIER fresh pins equal
# archived pins because the harness is byte-identical to its June state.
ROW_KEY_PIN = (
    "bias_sq", "block", "burnin", "chains", "config_index", "cost_trace_time",
    "coupling", "cuda_runtime_version", "cuda_visible_devices", "cupy_version",
    "dataset", "design_seed", "dim", "driver_version", "elapsed_sec",
    "elapsed_sec_per_ensemble", "eta", "git_hash", "gpu_name", "gpu_uuid",
    "hostname", "init_scale", "lin_chain_var", "lin_ens_var", "marginal_second",
    "method", "mode", "mse", "mse_to_single", "n_ensembles", "order_index",
    "order_seed", "prior", "ref_burnin", "ref_chains", "ref_reps", "ref_steps",
    "replicate", "reps", "script_hash", "seed", "single_bias_sq", "single_mse",
    "single_test_logloss", "single_trace_var", "steps", "target",
    "terminal_spread", "test_logloss", "test_logloss_var", "trace_var", "windows",
)
ROW_KEY_PIN_SET = frozenset(ROW_KEY_PIN)
DATASET_SEEDS = {"wdbc": 910000, "spambase": 920000, "adult": 930000}
FRESH_ORDER_SEED_BASE = 800000 - 40110 + 300  # order_seed = 800000 + rep - 40110 + 300 (AM-6)
FRESH_SMOKE = {
    "replicate": 40109,
    "dataset": "wdbc",
    "block": "smoke",
    "mode": "full",
    "init_scale": 0.0,
    "prior": 0.3,
    "seed": DATASET_SEEDS["wdbc"] + 10000 * 40109 + 0,  # --quick priors [0.3] => prior index 0
    "design_seed": DATASET_SEEDS["wdbc"] + 0,
    "order_seed": 800000 + 40109,  # registered smoke command passes no --order-seed
    "steps": 40,
    "burnin": 20,
    "ref_steps": 80,
    "ref_burnin": 40,
    "ref_reps": 8,
    "ref_chains": 8,
    "reps": 16,
    "n_ensembles": 16,
    "windows": 20,
    "rows": 10,
}
GPU_BY_BLOCK = {"primary": "A", "rand": "A", "smoke": "A", "stress": "B", "scale1": "B"}
K5_GPU_A_REPS = {40134, 40135, 40136}
ERROR_CAP = 40


def norm_host(value):
    """Prefix normalization: FQDN -> short host (identical to the archived
    host-prefix handling; VR-4).  Host names are redacted in the released rows."""
    return str(value).strip().split(".", 1)[0]


def expected_gpu_letter(block, rep):
    if block == "k5":
        return "A" if rep in K5_GPU_A_REPS else "B"
    return GPU_BY_BLOCK[block]


def expected_scoring_files(blocks):
    names = []
    for block in BLOCK_ORDER:
        for rep in blocks[block]["reps"]:
            names.append((f"{block}_rep{rep}.jsonl", block, rep))
    return names


def load_scoring_files(input_dir, blocks, errors, fresh):
    """Discover + load the scoring files; per-file structural checks."""
    expected = expected_scoring_files(blocks)
    expected_names = {name for name, _, _ in expected}
    present = sorted(p.name for p in input_dir.glob("*_rep*.jsonl"))
    if fresh:
        allowed_extra = {f"smoke_rep{FRESH_SMOKE['replicate']}.jsonl"}
        for name in present:
            if name not in expected_names and name not in allowed_extra:
                errors.append(f"[FILES] unexpected raw file (wrong block/replicate for fresh mode): {name}")
    rows = []
    rows_by_file = {}
    for name, block, rep in expected:
        path = input_dir / name
        if not path.exists():
            errors.append(f"[FILES] missing expected raw file: {path}")
            continue
        file_rows = load_rows([path])
        rows_by_file[name] = (block, rep, file_rows)
        expected_count = 18 if block == "k5" else 60
        if len(file_rows) != expected_count:
            errors.append(f"[FILES] {name}: {len(file_rows)} rows, expected {expected_count}")
        for i, row in enumerate(file_rows):
            if set(row.keys()) != ROW_KEY_PIN_SET:
                extra = sorted(set(row.keys()) - ROW_KEY_PIN_SET)
                miss = sorted(ROW_KEY_PIN_SET - set(row.keys()))
                errors.append(f"[SCHEMA] {name} row {i}: key set differs from the archived pin (extra={extra}, missing={miss})")
                continue
            if row["block"] != block:
                errors.append(f"[FILES] {name} row {i}: block field {row['block']!r} != {block!r}")
            if int(row["replicate"]) != rep:
                errors.append(f"[FILES] {name} row {i}: replicate {row['replicate']} != {rep}")
        rows.extend(file_rows)
    return rows, rows_by_file


def prior_index(prior):
    value = float(prior)
    for idx, ref in enumerate(PRIORS):
        if value == ref:
            return idx
    return None


def check_fresh_rows(rows_by_file, errors):
    """DR-4(d) + VR-4: exact seed/design_seed/order_seed recomputation and
    exact normalized-hostname equality on EVERY row of EVERY block."""
    count = 0
    for name in sorted(rows_by_file):
        block, rep, file_rows = rows_by_file[name]
        expected_order_seed = FRESH_ORDER_SEED_BASE + rep
        for i, row in enumerate(file_rows):
            if set(row.keys()) != ROW_KEY_PIN_SET:
                continue  # already reported by the schema check
            idx = prior_index(row["prior"])
            if idx is None:
                errors.append(f"[SEED] {name} row {i}: prior {row['prior']!r} not in {PRIORS}")
                continue
            expected_seed = DATASET_SEEDS.get(row["dataset"], None)
            if expected_seed is None:
                errors.append(f"[SEED] {name} row {i}: unknown dataset {row['dataset']!r}")
                continue
            expected_seed = expected_seed + 10000 * int(row["replicate"]) + idx
            if int(row["seed"]) != expected_seed:
                errors.append(f"[SEED] {name} row {i}: seed {row['seed']} != recomputed {expected_seed}")
            expected_design = DATASET_SEEDS[row["dataset"]] + idx
            if int(row["design_seed"]) != expected_design:
                errors.append(f"[SEED] {name} row {i}: design_seed {row['design_seed']} != recomputed {expected_design}")
            if int(row["order_seed"]) != expected_order_seed:
                errors.append(f"[SEED] {name} row {i}: order_seed {row['order_seed']} != recomputed {expected_order_seed}")
            if norm_host(row["hostname"]) != FRESH_HOST_SHORT:
                errors.append(f"[HOST] {name} row {i}: hostname {row['hostname']!r} != {FRESH_HOST_SHORT} (exact normalized equality, VR-4)")
            count += 1
    return count


def check_script_hashes(script_hashes, errors, where):
    if script_hashes != {EXPECTED_SCRIPT_HASH}:
        errors.append(f"[PROTOCOL] {where}: script_hash set {sorted(map(str, script_hashes))} != {{{EXPECTED_SCRIPT_HASH}}}")


def check_smoke_rows(smoke_rows, errors):
    name = f"smoke_rep{FRESH_SMOKE['replicate']}.jsonl"
    if len(smoke_rows) != FRESH_SMOKE["rows"]:
        errors.append(f"[SMOKE] {name}: {len(smoke_rows)} rows, expected {FRESH_SMOKE['rows']}")
    seen_configs = []
    for i, row in enumerate(smoke_rows):
        if set(row.keys()) != ROW_KEY_PIN_SET:
            extra = sorted(set(row.keys()) - ROW_KEY_PIN_SET)
            miss = sorted(ROW_KEY_PIN_SET - set(row.keys()))
            errors.append(f"[SMOKE] {name} row {i}: key set differs from the archived pin (extra={extra}, missing={miss})")
            continue
        for field in ["replicate", "dataset", "block", "mode", "init_scale", "prior", "seed", "design_seed",
                      "order_seed", "steps", "burnin", "ref_steps", "ref_burnin", "ref_reps", "ref_chains",
                      "reps", "n_ensembles", "windows"]:
            if row[field] != FRESH_SMOKE[field]:
                errors.append(f"[SMOKE] {name} row {i}: {field} {row[field]!r} != {FRESH_SMOKE[field]!r}")
        if row["script_hash"] != EXPECTED_SCRIPT_HASH:
            errors.append(f"[SMOKE] {name} row {i}: script_hash {row['script_hash']!r} != {EXPECTED_SCRIPT_HASH}")
        if norm_host(row["hostname"]) != FRESH_HOST_SHORT:
            errors.append(f"[SMOKE] {name} row {i}: hostname {row['hostname']!r} != {FRESH_HOST_SHORT}")
        if not finite_row(row):
            errors.append(f"[SMOKE] {name} row {i}: nonfinite value")
        if not metadata_present(row):
            errors.append(f"[SMOKE] {name} row {i}: metadata fields missing/empty")
        seen_configs.append((int(row["chains"]), row["method"]))
    if len(smoke_rows) == FRESH_SMOKE["rows"] and sorted(seen_configs) != sorted(FULL_CONFIGS):
        errors.append(f"[SMOKE] {name}: config set {sorted(seen_configs)} != full config list")


def parse_argv_flag(argv, flag):
    for i, token in enumerate(argv):
        token = str(token)
        if token == flag:
            if i + 1 < len(argv):
                return str(argv[i + 1])
            return None
        if token.startswith(flag + "="):
            return token.split("=", 1)[1]
    return None


def load_sidecars_for(path, sidecar_dir, errors):
    """Find <file>.launch<N>.json sidecars; ordinals must be 1..K contiguous."""
    directory = sidecar_dir if sidecar_dir is not None else path.parent
    pattern = re.compile(re.escape(path.name) + r"\.launch(\d+)\.json$")
    found = {}
    for candidate in sorted(directory.glob(path.name + ".launch*.json")):
        match = pattern.match(candidate.name)
        if not match:
            errors.append(f"[SIDECAR] {path.name}: malformed sidecar name {candidate.name}")
            continue
        ordinal = int(match.group(1))
        try:
            found[ordinal] = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"[SIDECAR] {candidate.name}: unreadable sidecar ({exc})")
    if not found:
        errors.append(f"[SIDECAR] {path.name}: no launch sidecar found (missing sidecar => file fails validation)")
        return None
    if sorted(found) != list(range(1, len(found) + 1)):
        errors.append(f"[SIDECAR] {path.name}: launch ordinals {sorted(found)} not contiguous from 1")
        return None
    return found


def check_file_sidecars(path, block, rep, file_rows, sidecar_dir, gpu_uuids, errors):
    """VR-1(f)/VR-5 row-vs-sidecar assertions + VR-4 block->GPU map."""
    sidecars = load_sidecars_for(path, sidecar_dir, errors)
    if sidecars is None:
        return
    expected_uuid = gpu_uuids[expected_gpu_letter(block, rep)]
    by_dataset = {}
    for ordinal in sorted(sidecars):
        sc = sidecars[ordinal]
        tag = f"{path.name}.launch{ordinal}.json"
        if sc.get("schema") != "regen-sidecar-1":
            errors.append(f"[SIDECAR] {tag}: schema {sc.get('schema')!r} != 'regen-sidecar-1'")
        if sc.get("status") != "completed" or sc.get("exit_code") != 0:
            errors.append(f"[SIDECAR] {tag}: status={sc.get('status')!r} exit_code={sc.get('exit_code')!r} (must be completed/0)")
        argv = sc.get("argv")
        if not isinstance(argv, list):
            errors.append(f"[SIDECAR] {tag}: argv missing/not a list")
            continue
        dataset = parse_argv_flag(argv, "--dataset")
        if dataset is None:
            errors.append(f"[SIDECAR] {tag}: argv carries no --dataset")
            continue
        if dataset in by_dataset:
            errors.append(f"[SIDECAR] {tag}: duplicate sidecar for dataset {dataset}")
            continue
        by_dataset[dataset] = (tag, sc)
        argv_rep = parse_argv_flag(argv, "--replicate")
        if argv_rep is None or int(argv_rep) != rep:
            errors.append(f"[SIDECAR] {tag}: argv --replicate {argv_rep!r} != {rep}")
        if norm_host(sc.get("hostname", "")) != FRESH_HOST_SHORT:
            errors.append(f"[SIDECAR] {tag}: hostname {sc.get('hostname')!r} != {FRESH_HOST_SHORT}")
        if sc.get("gpu_uuid") != expected_uuid:
            errors.append(
                f"[SIDECAR] {tag}: gpu_uuid {sc.get('gpu_uuid')!r} != registered "
                f"GPU_{expected_gpu_letter(block, rep)} uuid {expected_uuid!r} for block {block} rep {rep}"
            )
    row_datasets = sorted({row.get("dataset") for row in file_rows})
    if sorted(by_dataset) != row_datasets:
        errors.append(f"[SIDECAR] {path.name}: sidecar datasets {sorted(by_dataset)} != row datasets {row_datasets}")
    for i, row in enumerate(file_rows):
        entry = by_dataset.get(row.get("dataset"))
        if entry is None:
            continue
        tag, sc = entry
        hashes = {p: h for p, h in (sc.get("script_sha256") or {}).items() if Path(p).name == HARNESS_BASENAME}
        if not hashes:
            errors.append(f"[SIDECAR] {tag}: script_sha256 has no entry for {HARNESS_BASENAME}")
        else:
            digests = {str(h)[:16] for h in hashes.values()}
            if len(digests) != 1 or row.get("script_hash") not in digests:
                errors.append(f"[SIDECAR] {path.name} row {i}: row script_hash {row.get('script_hash')!r} != sidecar sha256(harness)[:16] {sorted(digests)}")
        if norm_host(row.get("hostname", "")) != norm_host(sc.get("hostname", "")):
            errors.append(f"[SIDECAR] {path.name} row {i}: row hostname {row.get('hostname')!r} != sidecar hostname {sc.get('hostname')!r}")
        if row.get("gpu_uuid") != sc.get("gpu_uuid"):
            errors.append(f"[SIDECAR] {path.name} row {i}: row gpu_uuid {row.get('gpu_uuid')!r} != sidecar gpu_uuid {sc.get('gpu_uuid')!r}")
        git_hash = str(row.get("git_hash", "")).strip()
        git_head = str(sc.get("git_head", "")).strip()
        if not git_hash:
            errors.append(f"[SIDECAR] {path.name} row {i}: row git_hash empty")
        elif not git_head.startswith(git_hash):
            # VR-5/VR-8: the ONLY registered git comparison (abbreviated form).
            errors.append(f"[SIDECAR] {path.name} row {i}: sidecar git_head {git_head!r} does not start with row git_hash {git_hash!r}")


# ---------------------------------------------------------------------------
# Per-replicate ratio details for the DR-7 comparison record.  Re-derives the
# exact per-replicate ratios of the cloned pipeline (same get_row/cost).
# ---------------------------------------------------------------------------

def cell_details(rows_by_key, spec, dataset, prior):
    details = {"high": {}, "odd": {}, "odd_comparator": {}}
    for rep in spec["reps"]:
        if spec["mode"] == "full":
            z8 = get_row(rows_by_key, dataset, prior, 8, "zero_sum", rep, spec["scale"], "full")
            z7 = get_row(rows_by_key, dataset, prior, 7, "zero_sum", rep, spec["scale"], "full")
            a7 = get_row(rows_by_key, dataset, prior, 7, "antithetic", rep, spec["scale"], "full")
            iid7 = get_row(rows_by_key, dataset, prior, 7, "iid", rep, spec["scale"], "full")
            best_cost = min(
                cost(get_row(rows_by_key, dataset, prior, chains, "iid", rep, spec["scale"], "full"))
                for chains in FULL_IID_K
            )
            details["high"][rep] = cost(z8) / best_cost
            cost_iid, cost_ant = cost(iid7), cost(a7)
            details["odd"][rep] = cost(z7) / min(cost_iid, cost_ant)
            details["odd_comparator"][rep] = "iid" if cost_iid <= cost_ant else "antithetic"
        else:
            z5 = get_row(rows_by_key, dataset, prior, 5, "zero_sum", rep, spec["scale"], "k5")
            a5 = get_row(rows_by_key, dataset, prior, 5, "antithetic", rep, spec["scale"], "k5")
            iid5 = get_row(rows_by_key, dataset, prior, 5, "iid", rep, spec["scale"], "k5")
            cost_iid, cost_ant = cost(iid5), cost(a5)
            details["odd"][rep] = cost(z5) / min(cost_iid, cost_ant)
            details["odd_comparator"][rep] = "iid" if cost_iid <= cost_ant else "antithetic"
    return details


def winning_component(cell):
    high = cell["high"]
    return "high" if (np.isfinite(high) and high > cell["odd"]) else "odd"


def numeric_comparison(frozen_value, fresh_values_by_rep):
    values = [fresh_values_by_rep[rep] for rep in sorted(fresh_values_by_rep)]
    fresh_med = median(values)
    fresh_min = min(values)
    fresh_max = max(values)
    in_min_max = fresh_min <= frozen_value <= fresh_max
    rel_dev = abs(fresh_med - frozen_value) / abs(frozen_value)
    rel_within = rel_dev <= 0.15
    if in_min_max:
        verdict, criterion = "in", "frozen-in-fresh-min-max"
    elif rel_within:
        verdict, criterion = "in", "15pct-relative-fallback"
    else:
        verdict, criterion = "out", "none"
    return {
        "fresh_median": float(fresh_med),
        "fresh_min": float(fresh_min),
        "fresh_max": float(fresh_max),
        "fresh_per_replicate": {str(rep): float(fresh_values_by_rep[rep]) for rep in sorted(fresh_values_by_rep)},
        "in_fresh_min_max": bool(in_min_max),
        "rel_dev_15pct": {"value": float(rel_dev), "within": bool(rel_within)},
        "verdict": verdict,
        "criterion": criterion,
    }


def build_comparison_record(frozen_cells_by_block, fresh_cells_by_block, frozen_keyed, fresh_keyed, args, fresh_summary, gate_info):
    """DR-7(d,e): exactly 14 numeric outcomes (4 high + 5 odd + 5 block maxima),
    5 worst-cell identity verdicts, descriptive best-odd-K comparator identities."""
    numeric_outcomes = []
    worst_cell_identities = []
    comparator_identities = []
    component_entries = {}

    for block in BLOCK_ORDER:
        frozen_cells = frozen_cells_by_block[block]
        fresh_cells = fresh_cells_by_block[block]
        frozen_spec = ARCHIVED_BLOCKS[block]
        fresh_spec = FRESH_BLOCKS[block]
        frozen_worst = worst_cell_of(frozen_cells)
        fresh_worst = worst_cell_of(fresh_cells)
        dataset, prior = frozen_worst["dataset"], frozen_worst["prior"]
        frozen_det = cell_details(frozen_keyed, frozen_spec, dataset, prior)
        fresh_det = cell_details(fresh_keyed, fresh_spec, dataset, prior)

        kinds = ["odd"] if block == "k5" else ["high", "odd"]
        for kind in kinds:
            comparison = numeric_comparison(frozen_worst[kind], fresh_det[kind])
            entry = {
                "outcome_id": f"{block}/{kind}",
                "block": block,
                "kind": kind,
                "cell": {"dataset": dataset, "prior": prior, "label": cell_label(frozen_worst)},
                "frozen_value": float(frozen_worst[kind]),
                "frozen_printed": f"{frozen_worst[kind]:.4f}",
            }
            entry.update(comparison)
            numeric_outcomes.append(entry)
            component_entries[(block, kind)] = entry

    for block in BLOCK_ORDER:
        frozen_cells = frozen_cells_by_block[block]
        fresh_cells = fresh_cells_by_block[block]
        frozen_worst = worst_cell_of(frozen_cells)
        fresh_worst = worst_cell_of(fresh_cells)
        frozen_component = winning_component(frozen_worst)
        fresh_component = winning_component(fresh_worst)
        frozen_score = worst(frozen_cells, "score")
        fresh_score = worst(fresh_cells, "score")
        inherited = component_entries[(block, frozen_component)]
        argmax_equal = (
            fresh_worst["dataset"] == frozen_worst["dataset"]
            and fresh_worst["prior"] == frozen_worst["prior"]
            and fresh_component == frozen_component
        )
        entry = {
            "outcome_id": f"{block}/block_max",
            "block": block,
            "kind": "block_max",
            "frozen_value": float(frozen_score),
            "frozen_printed": f"{frozen_score:.4f}",
            "frozen_winner": {
                "dataset": frozen_worst["dataset"],
                "prior": frozen_worst["prior"],
                "label": cell_label(frozen_worst),
                "component": frozen_component,
            },
            # DR-7(d): block maxima inherit the winning component cell's verdict.
            "verdict": inherited["verdict"],
            "verdict_inherited_from": inherited["outcome_id"],
            "frozen_identity_comparison": {
                key: inherited[key]
                for key in ["fresh_median", "fresh_min", "fresh_max", "in_fresh_min_max", "rel_dev_15pct", "criterion"]
            },
            "fresh_block_max": float(fresh_score),
            "fresh_argmax": {
                "dataset": fresh_worst["dataset"],
                "prior": fresh_worst["prior"],
                "label": cell_label(fresh_worst),
                "component": fresh_component,
            },
            "argmax_equal_to_frozen": bool(argmax_equal),
        }
        numeric_outcomes.append(entry)

        worst_cell_identities.append(
            {
                "block": block,
                "frozen": cell_label(frozen_worst),
                "fresh": cell_label(fresh_worst),
                "verdict": "equal" if cell_label(fresh_worst) == cell_label(frozen_worst) else "not-equal",
            }
        )

        # DR-7(e): descriptive only, never a match verdict.
        frozen_spec = ARCHIVED_BLOCKS[block]
        fresh_spec = FRESH_BLOCKS[block]
        cells_to_report = [(frozen_worst["dataset"], frozen_worst["prior"], "frozen worst cell")]
        if (fresh_worst["dataset"], fresh_worst["prior"]) != (frozen_worst["dataset"], frozen_worst["prior"]):
            cells_to_report.append((fresh_worst["dataset"], fresh_worst["prior"], "fresh argmax cell"))
        for dataset, prior, role in cells_to_report:
            frozen_det = cell_details(frozen_keyed, frozen_spec, dataset, prior)
            fresh_det = cell_details(fresh_keyed, fresh_spec, dataset, prior)
            comparator_identities.append(
                {
                    "block": block,
                    "role": role,
                    "cell": {"dataset": dataset, "prior": prior},
                    "frozen_per_replicate": {str(rep): frozen_det["odd_comparator"][rep] for rep in sorted(frozen_det["odd_comparator"])},
                    "fresh_per_replicate": {str(rep): fresh_det["odd_comparator"][rep] for rep in sorted(fresh_det["odd_comparator"])},
                    "note": "descriptive per DR-7(e); no match verdict",
                }
            )

    n_numeric = len([e for e in numeric_outcomes])
    assert n_numeric == 14, f"expected 14 numeric outcomes, built {n_numeric}"

    record = {
        "schema": "regen-frontier-comparison-1",
        "table": "campaign_wallclock_frontier",
        "generator": "scripts/regen_frontier_table.py",
        "frozen_source": str(args.archived_dir),
        "fresh_source": str(args.input_dir),
        "gpu_a_uuid": args.gpu_a_uuid,
        "gpu_b_uuid": args.gpu_b_uuid,
        "match_rule": (
            "DR-7 completed AM-13 rule (result-review corrected 2026-07-19): every FRONTIER numeric outcome has "
            "underlying per-replicate values, so the fresh min-max clause is the registered verdict; the 15% "
            "relative quantity (DR-7b) is a descriptive sensitivity field only, never a pass/fail fallback (all "
            "14 outcomes here pass min-max, so no verdict ever depended on it); block maxima inherit the winning component "
            "cell's verdict with fresh argmax disclosed on change (DR-7d); worst-cell identities get "
            "equal/not-equal verdicts (DR-7e); per-replicate best-odd-K comparator identities are descriptive "
            "with no match verdict (DR-7e); pass fractions are descriptive only, never calibrated statistics (DR-8)."
        ),
        "numeric_outcome_count": {"high_ratios": 4, "odd_ratios": 5, "block_maxima": 5, "total": 14},
        "archived_byte_gates": gate_info,
        "disclosures": {
            "am0": (
                "AM-0 facts correction: the archived attempt08 raws, runner (sha256[:16] "
                + EXPECTED_SCRIPT_HASH
                + ") and analyzer survive in-tree and the frozen values recompute exactly on CPU; this fresh-seed "
                "block is the registered confirmation layer, not a restoration."
            ),
            "am5": "printed Replicates column reads 40110-40139 instead of 300-329 (no-code-change seed mapping).",
            "am6": "order seeds 800300-800329 reuse the ORIGINAL config-order shuffles; they affect run order only, never sampling RNG.",
            "dr5": (
                "same-host cross-device stress register: the fresh stress block ran on the same host ("
                + FRESH_HOST_SHORT
                + ") on a distinct GPU (GPU_B); concordance with the frozen cross-node stress block is a "
                "cross-device robustness result, never a like-for-like cross-node replay."
            ),
            "seed_sharing_vr2": {
                "intentional_same_design": (
                    "matched-initialization trio iid/zero_sum/antithetic at the same dataset/replicate/prior/K "
                    "shares the init-stream seed seed+60617+1009*K (e.g. 402077680 at wdbc rep 40110 prior 0.1 K=7); "
                    "reference stream seed+77717 and probe design_seed+1729 are shared across methods within a cell "
                    "(AM-9: probe design_seed is replicate-independent, outside the 402M block, diagnostics only)."
                ),
                "accidental_cross_target": (
                    "adjacent-dataset/adjacent-replicate base-seed coincidences across targets, e.g. base 402020000 "
                    "shared by wdbc rep 40111 and spambase rep 40110 (DR-3/VR-2); same inherited structure as the "
                    "June originals; the four campaign envelopes remain disjoint."
                ),
            },
        },
        "numeric_outcomes": numeric_outcomes,
        "worst_cell_identities": worst_cell_identities,
        "best_odd_comparator_identities": comparator_identities,
        "fresh_empirical_gates": fresh_summary,
    }
    return record


# ---------------------------------------------------------------------------
# Mode drivers.
# ---------------------------------------------------------------------------

def fail(errors):
    print(f"FAIL-CLOSED: {len(errors)} validation error(s); no output written.", file=sys.stderr)
    by_category = {}
    for err in errors:
        category = err.split("]", 1)[0] + "]" if err.startswith("[") else "[OTHER]"
        by_category.setdefault(category, []).append(err)
    for category in sorted(by_category):
        entries = by_category[category]
        for err in entries[:ERROR_CAP]:
            print("  " + err, file=sys.stderr)
        if len(entries) > ERROR_CAP:
            print(f"  ... {len(entries) - ERROR_CAP} more {category} error(s)", file=sys.stderr)
    sys.exit(1)


def first_divergence(actual, expected):
    limit = min(len(actual), len(expected))
    for i in range(limit):
        if actual[i] != expected[i]:
            return i
    return limit if len(actual) != len(expected) else -1


def analyze(input_dir, blocks, errors, fresh):
    """Load + validate + compute cells (shared by both modes)."""
    rows, rows_by_file = load_scoring_files(input_dir, blocks, errors, fresh=fresh)
    if errors:
        # File-level integrity failed; the cell pipeline would be unreliable.
        # Still surface fresh row-level violations on whatever loaded (diagnostics).
        if fresh:
            check_fresh_rows(rows_by_file, errors)
        return None
    rows_by_key, duplicates, nonfinite, protocol_errors, script_hashes = validate_rows(rows)
    check_script_hashes(script_hashes, errors, "scoring rows")
    missing = []
    host_errors = []
    cells_by_block = {}
    for name, spec in blocks.items():
        block_missing, block_host_errors = required_missing(rows_by_key, name, spec)
        missing.extend(block_missing)
        host_errors.extend(block_host_errors)
        if not block_missing:
            if spec["mode"] == "full":
                cells_by_block[name] = full_block_cells(rows_by_key, name, spec)
            else:
                cells_by_block[name] = k5_block_cells(rows_by_key, name, spec)
        else:
            cells_by_block[name] = []
    if fresh:
        check_fresh_rows(rows_by_file, errors)
    for item in duplicates[:ERROR_CAP]:
        errors.append(f"[PROTOCOL] duplicate row key: {item!r}")
    for item in nonfinite[:ERROR_CAP]:
        errors.append(f"[PROTOCOL] nonfinite row: {item!r}")
    for key, failed in protocol_errors[:ERROR_CAP]:
        errors.append(f"[PROTOCOL] {key!r}: failed checks {failed}")
    for item in missing[:ERROR_CAP]:
        errors.append(f"[PROTOCOL] missing required row: {item!r}")
    for item in host_errors[:ERROR_CAP]:
        errors.append(f"[HOST] host prefix violation: {item!r}")
    if len(duplicates) > ERROR_CAP or len(nonfinite) > ERROR_CAP or len(protocol_errors) > ERROR_CAP or len(missing) > ERROR_CAP or len(host_errors) > ERROR_CAP:
        errors.append("[PROTOCOL] additional errors truncated")
    return {
        "rows": rows,
        "rows_by_file": rows_by_file,
        "rows_by_key": rows_by_key,
        "duplicates": duplicates,
        "nonfinite": nonfinite,
        "protocol_errors": protocol_errors,
        "script_hashes": script_hashes,
        "missing": missing,
        "host_errors": host_errors,
        "cells_by_block": cells_by_block,
    }


def check_archived_values(cells_by_block, errors):
    for block in BLOCK_ORDER:
        cells = cells_by_block[block]
        cell = worst_cell_of(cells)
        expected_high, expected_odd, expected_tag = ARCHIVED_CHECK_VALUES[block]
        got_high = "--" if block == "k5" else f"{cell['high']:.4f}"
        got_odd = f"{cell['odd']:.4f}"
        got_tag = "K=5" if block == "k5" else "K=7"
        got_cell = cell_label(cell)
        if (got_high, got_odd, got_tag, got_cell) != (expected_high, expected_odd, expected_tag, ARCHIVED_WORST_CELL):
            errors.append(
                f"[CHECK-VALUES] {block}: got ({got_high}, {got_odd} ({got_tag}), {got_cell}) "
                f"!= registered ({expected_high}, {expected_odd} ({expected_tag}), {ARCHIVED_WORST_CELL})"
            )


def run_archived_pipeline(archived_dir):
    """Full archived-mode computation + DR-4b byte gates (in memory).
    Returns (result, errors); result carries md/tex bytes and cells."""
    errors = []
    analysis = analyze(archived_dir, ARCHIVED_BLOCKS, errors, fresh=False)
    if errors or analysis is None:
        return None, errors
    md_text, summary = build_result_md(
        analysis["rows"],
        analysis["cells_by_block"],
        ARCHIVED_BLOCKS,
        analysis["script_hashes"],
        analysis["missing"],
        analysis["duplicates"],
        analysis["nonfinite"],
        analysis["protocol_errors"],
        analysis["host_errors"],
        ARCHIVED_MD_LABELS,
    )
    tex_text = build_tex(analysis["cells_by_block"], ARCHIVED_BLOCKS, ARCHIVED_TEX_LABELS)
    check_archived_values(analysis["cells_by_block"], errors)
    reparse_tex(tex_text, analysis["cells_by_block"], ARCHIVED_BLOCKS, ARCHIVED_TEX_LABELS, errors)

    md_bytes = md_text.encode("utf-8")
    tex_bytes = tex_text.encode("utf-8")
    frozen_md_bytes = FROZEN_MD.read_bytes()
    frozen_tex_bytes = FROZEN_TEX.read_bytes()
    if md_bytes != frozen_md_bytes:
        pos = first_divergence(md_bytes, frozen_md_bytes)
        errors.append(
            f"[GATE-1] result markdown NOT byte-identical to {FROZEN_MD} "
            f"(len {len(md_bytes)} vs {len(frozen_md_bytes)}; first divergence at byte {pos}: "
            f"emitted {md_bytes[pos:pos+40]!r} vs frozen {frozen_md_bytes[pos:pos+40]!r})"
        )
    if tex_bytes != frozen_tex_bytes:
        pos = first_divergence(tex_bytes, frozen_tex_bytes)
        errors.append(
            f"[GATE-2] .tex NOT byte-identical to {FROZEN_TEX} "
            f"(len {len(tex_bytes)} vs {len(frozen_tex_bytes)}; first divergence at byte {pos}: "
            f"emitted {tex_bytes[pos:pos+40]!r} vs frozen {frozen_tex_bytes[pos:pos+40]!r})"
        )
    result = {
        "analysis": analysis,
        "md_bytes": md_bytes,
        "tex_bytes": tex_bytes,
        "frozen_md_sha256": hashlib.sha256(frozen_md_bytes).hexdigest(),
        "frozen_tex_sha256": hashlib.sha256(frozen_tex_bytes).hexdigest(),
        "md_sha256": hashlib.sha256(md_bytes).hexdigest(),
        "tex_sha256": hashlib.sha256(tex_bytes).hexdigest(),
        "summary": summary,
    }
    return result, errors


def print_check_values(cells_by_block):
    print("Archived-mode check values (registered):")
    for block in BLOCK_ORDER:
        cell = worst_cell_of(cells_by_block[block])
        high_txt = "--" if block == "k5" else f"{cell['high']:.4f}"
        tag = "K=5" if block == "k5" else "K=7"
        print(f"  {block:8s} worst cell {cell_label(cell)}: high={high_txt} odd={cell['odd']:.4f} (${tag}$)")


def run_archived(args):
    out_dir = args.out_dir
    if out_dir is None:
        print("--mode archived requires --out-dir (scratch destination for the byte-gated artifacts)", file=sys.stderr)
        sys.exit(2)
    out_dir = Path(out_dir).resolve()
    for forbidden in [REPO / "paper", REPO / "campaigns"]:
        if out_dir == forbidden or forbidden in out_dir.parents:
            print(f"refusing --out-dir under {forbidden} (frozen artifacts must never be overwritten)", file=sys.stderr)
            sys.exit(2)
    result, errors = run_archived_pipeline(args.input_dir)
    if errors or result is None:
        fail(errors)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / FROZEN_MD.name
    out_tex = out_dir / FROZEN_TEX.name
    out_md.write_bytes(result["md_bytes"])
    out_tex.write_bytes(result["tex_bytes"])
    print("GATE 1 PASS: analyzer result markdown byte-identical")
    print(f"  emitted {out_md}")
    print(f"    sha256 {result['md_sha256']}")
    print(f"  frozen  {FROZEN_MD}")
    print(f"    sha256 {result['frozen_md_sha256']}")
    print("GATE 2 PASS: worst-cell .tex byte-identical")
    print(f"  emitted {out_tex}")
    print(f"    sha256 {result['tex_sha256']}")
    print(f"  frozen  {FROZEN_TEX}")
    print(f"    sha256 {result['frozen_tex_sha256']}")
    print_check_values(result["analysis"]["cells_by_block"])
    print("ARCHIVED MODE: ALL GATES PASS")


def run_fresh(args):
    if args.out_dir is not None:
        print("--out-dir is not accepted in fresh mode: outputs are pinned to "
              f"{FRESH_TABLES_DIR} by DR-4(e)", file=sys.stderr)
        sys.exit(2)
    errors = []
    if not args.gpu_a_uuid or not args.gpu_b_uuid:
        errors.append("[ARGS] fresh mode requires --gpu-a-uuid and --gpu-b-uuid (registered block->GPU map, VR-4)")
    elif args.gpu_a_uuid == args.gpu_b_uuid:
        errors.append("[ARGS] GPU_A uuid must differ from GPU_B uuid (VR-4)")

    # DR-4(b): the generator must reproduce the frozen artifacts from the
    # archived raws before being trusted on fresh raws.
    archived_result, archived_errors = run_archived_pipeline(args.archived_dir)
    if archived_errors or archived_result is None:
        print("archived byte gates FAILED; the generator may not be trusted on fresh raws", file=sys.stderr)
        fail(archived_errors)

    analysis = analyze(args.input_dir, FRESH_BLOCKS, errors, fresh=True)
    smoke_path = args.input_dir / f"smoke_rep{FRESH_SMOKE['replicate']}.jsonl"
    smoke_rows = None
    if smoke_path.exists():
        smoke_rows = load_rows([smoke_path])
        check_smoke_rows(smoke_rows, errors)
    if analysis is not None:
        gpu_uuids = {"A": args.gpu_a_uuid, "B": args.gpu_b_uuid}
        if args.gpu_a_uuid and args.gpu_b_uuid and args.gpu_a_uuid != args.gpu_b_uuid:
            for name in sorted(analysis["rows_by_file"]):
                block, rep, file_rows = analysis["rows_by_file"][name]
                check_file_sidecars(args.input_dir / name, block, rep, file_rows, args.sidecar_dir, gpu_uuids, errors)
            if smoke_rows is not None:
                check_file_sidecars(smoke_path, "smoke", FRESH_SMOKE["replicate"], smoke_rows, args.sidecar_dir, gpu_uuids, errors)
        else:
            errors.append("[SIDECAR] sidecar block->GPU validation skipped: gpu uuids not supplied (fail-closed)")
    if errors or analysis is None:
        fail(errors)

    fresh_md_text, fresh_summary = build_result_md(
        analysis["rows"],
        analysis["cells_by_block"],
        FRESH_BLOCKS,
        analysis["script_hashes"],
        analysis["missing"],
        analysis["duplicates"],
        analysis["nonfinite"],
        analysis["protocol_errors"],
        analysis["host_errors"],
        FRESH_MD_LABELS,
    )
    fresh_tex_text = build_tex(analysis["cells_by_block"], FRESH_BLOCKS, FRESH_TEX_LABELS)
    reparse_tex(fresh_tex_text, analysis["cells_by_block"], FRESH_BLOCKS, FRESH_TEX_LABELS, errors)
    if errors:
        fail(errors)

    gate_info = {
        "result_md": "PASS (byte-identical in this run)",
        "tex": "PASS (byte-identical in this run)",
        "frozen_md_sha256": archived_result["frozen_md_sha256"],
        "frozen_tex_sha256": archived_result["frozen_tex_sha256"],
    }
    record = build_comparison_record(
        archived_result["analysis"]["cells_by_block"],
        analysis["cells_by_block"],
        archived_result["analysis"]["rows_by_key"],
        analysis["rows_by_key"],
        args,
        fresh_summary,
        gate_info,
    )

    FRESH_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = FRESH_TABLES_DIR / "campaign_wallclock_frontier_fresh.tex"
    out_json = FRESH_TABLES_DIR / "comparison_frontier.json"
    for path in [out_tex, out_json]:
        if path.exists():
            print(f"refusing to overwrite existing fresh output {path} (delete it first if a re-run is intended)", file=sys.stderr)
            sys.exit(1)
    out_tex.write_text(fresh_tex_text, encoding="utf-8")
    def _relpaths(o):
        if isinstance(o, dict):
            return {_relpaths(k): _relpaths(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_relpaths(v) for v in o]
        if isinstance(o, str) and o.startswith(str(REPO) + "/"):
            return o[len(str(REPO)) + 1:]
        return o
    record = _relpaths(record)
    out_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(fresh_md_text)
    print(f"fresh .tex written: {out_tex}")
    print(f"  sha256 {hashlib.sha256(fresh_tex_text.encode('utf-8')).hexdigest()}")
    print(f"comparison record written: {out_json}")
    verdicts = [(entry["outcome_id"], entry["verdict"]) for entry in record["numeric_outcomes"]]
    n_in = sum(1 for _, verdict in verdicts if verdict == "in")
    print(f"numeric outcomes in/out: {n_in}/{len(verdicts)} in -- " + ", ".join(f"{oid}={v}" for oid, v in verdicts))
    identity = ", ".join(f"{e['block']}={e['verdict']}" for e in record["worst_cell_identities"])
    print(f"worst-cell identities: {identity}")
    print("FRESH MODE: validation passed; outputs emitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["archived", "fresh"], required=True)
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="raw JSONL directory (default: archived -> campaigns/attempt08, "
                             "fresh -> outputs/sweeps/camera_ready_regen/frontier)")
    parser.add_argument("--archived-dir", type=Path, default=DEFAULT_ARCHIVED_DIR,
                        help="archived raws used for the frozen full-precision comparanda in fresh mode")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="archived mode only: scratch directory for the byte-gated artifacts")
    parser.add_argument("--sidecar-dir", type=Path, default=None,
                        help="fresh mode: directory holding <file>.launch<N>.json sidecars "
                             "(default: each input file's own directory)")
    parser.add_argument("--gpu-a-uuid", default=None, help="fresh mode: registered GPU_A uuid")
    parser.add_argument("--gpu-b-uuid", default=None, help="fresh mode: registered GPU_B uuid")
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = DEFAULT_ARCHIVED_DIR if args.mode == "archived" else DEFAULT_FRESH_DIR
    if args.mode == "archived":
        run_archived(args)
    else:
        run_fresh(args)


if __name__ == "__main__":
    main()
