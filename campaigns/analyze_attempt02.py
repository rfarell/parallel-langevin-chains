#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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


def analyze(paths, chains):
    rows = load_rows(paths)
    by_key = defaultdict(list)
    for row in rows:
        if int(row["chains"]) != chains:
            continue
        by_key[(row["dataset"], float(row["prior"]), row["method"])].append(row)

    cells = []
    missing = []
    for dataset in ["wdbc", "spambase", "adult"]:
        for prior in [0.1, 0.3]:
            z_rows = by_key.get((dataset, prior, "zero_sum"), [])
            a_rows = by_key.get((dataset, prior, "antithetic"), [])
            i_rows = by_key.get((dataset, prior, "iid"), [])
            z_by_rep = {row["replicate"]: row for row in z_rows}
            a_by_rep = {row["replicate"]: row for row in a_rows}
            i_by_rep = {row["replicate"]: row for row in i_rows}
            reps = sorted(set(z_by_rep) & set(a_by_rep) & set(i_by_rep))
            if len(reps) != min(len(z_rows), len(a_rows), len(i_rows)) or not reps:
                missing.append((dataset, prior, len(z_rows), len(a_rows), len(i_rows), len(reps)))
                continue
            cells.append(
                {
                    "dataset": dataset,
                    "prior": prior,
                    "replicates": reps,
                    "zero_trace": median([z_by_rep[rep]["trace_var"] for rep in reps]),
                    "ant_trace": median([a_by_rep[rep]["trace_var"] for rep in reps]),
                    "iid_trace": median([i_by_rep[rep]["trace_var"] for rep in reps]),
                    "zero_over_ant": median(
                        [z_by_rep[rep]["trace_var"] / a_by_rep[rep]["trace_var"] for rep in reps]
                    ),
                    "zero_over_iid": median(
                        [z_by_rep[rep]["trace_var"] / i_by_rep[rep]["trace_var"] for rep in reps]
                    ),
                    "ant_over_iid": median(
                        [a_by_rep[rep]["trace_var"] / i_by_rep[rep]["trace_var"] for rep in reps]
                    ),
                    "zero_mse_over_ant": median([z_by_rep[rep]["mse"] / a_by_rep[rep]["mse"] for rep in reps]),
                    "zero_logloss_over_ant": median(
                        [z_by_rep[rep]["test_logloss"] / a_by_rep[rep]["test_logloss"] for rep in reps]
                    ),
                    "zero_sec_over_ant": median(
                        [
                            z_by_rep[rep]["elapsed_sec_per_ensemble"]
                            / a_by_rep[rep]["elapsed_sec_per_ensemble"]
                            for rep in reps
                        ]
                    ),
                }
            )
    return cells, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--chains", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    cells, missing = analyze(args.jsonl, args.chains)
    n_odd = max((cell["zero_over_ant"] for cell in cells), default=float("nan"))
    floor = 0.20
    title = args.title or f"Attempt 2 K={args.chains} Odd-K Result"

    lines = [
        f"# {title}",
        "",
        "## Pre-Committed Metric",
        "",
        (
            "N_odd = max over dataset/prior of median matched-replicate "
            f"trace_var(zero_sum,K={args.chains}) / trace_var(antithetic,K={args.chains})."
        ),
        f"Material floor: N_odd <= {floor:.2f}.",
        f"Observed N_odd: {fmt(n_odd)}.",
        "",
        "## Cell Results",
        "",
        "| dataset | prior | reps | zero trace | odd-ant trace | iid trace | zero/ant | zero/iid | ant/iid | zero/ant MSE | zero/ant logloss | zero/ant sec/ens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        lines.append(
            "| {dataset} | {prior:.1f} | {reps} | {zero_trace} | {ant_trace} | {iid_trace} | "
            "{zero_ant} | {zero_iid} | {ant_iid} | {zero_mse_ant} | {zero_logloss_ant} | "
            "{zero_sec_ant} |".format(
                dataset=cell["dataset"],
                prior=cell["prior"],
                reps=",".join(str(rep) for rep in cell["replicates"]),
                zero_trace=fmt(cell["zero_trace"]),
                ant_trace=fmt(cell["ant_trace"]),
                iid_trace=fmt(cell["iid_trace"]),
                zero_ant=fmt(cell["zero_over_ant"]),
                zero_iid=fmt(cell["zero_over_iid"]),
                ant_iid=fmt(cell["ant_over_iid"]),
                zero_mse_ant=fmt(cell["zero_mse_over_ant"]),
                zero_logloss_ant=fmt(cell["zero_logloss_over_ant"]),
                zero_sec_ant=fmt(cell["zero_sec_over_ant"]),
            )
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- Max-cell odd-K trace-variance metric clears floor: {'yes' if n_odd <= floor else 'no'}.",
            "- MSE/log-loss columns are guardrails only and do not rescue a failed trace-variance metric.",
        ]
    )
    if missing:
        lines.append("")
        lines.append("## Missing Or Unmatched Cells")
        for dataset, prior, z_count, a_count, i_count, rep_count in missing:
            lines.append(
                f"- {dataset} prior {prior}: zero_sum={z_count}, antithetic={a_count}, "
                f"iid={i_count}, matched_reps={rep_count}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
