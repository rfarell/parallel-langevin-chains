import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def fmt(x):
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.3f}"


def make_table(rows):
    targets = ["gaussian", "quartic", "logistic"]
    methods = ["iid", "zero_sum", "antithetic", "common"]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Target & iid & $C_\\star$ & pairwise & common \\\\",
        "\\midrule",
    ]
    for target in targets:
        vals = []
        for method in methods:
            ratios = [
                max(row["empirical_ratio_to_chain"], 1e-15)
                for row in rows
                if row["target"] == target and row["method"] == method
                and "init_mode" not in row
            ]
            vals.append(fmt(float(np.median(ratios))))
        lines.append(f"{target} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_init_table(rows):
    init_modes = ["common_zero", "zero_sum_random", "iid_random", "common_random"]
    labels = {
        "common_zero": "common zero",
        "zero_sum_random": "zero-sum random",
        "iid_random": "iid random",
        "common_random": "common random",
    }
    methods = ["iid", "zero_sum", "common"]
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Steps & Initialization & iid & $C_\\star$ & common \\\\",
        "\\midrule",
    ]
    init_rows = [row for row in rows if row.get("target") == "gaussian" and "init_mode" in row]
    for steps in sorted({row["steps"] for row in init_rows}):
        for mode in init_modes:
            vals = []
            for method in methods:
                ratios = [
                    max(row["empirical_ratio_to_chain"], 1e-15)
                    for row in init_rows
                    if row["steps"] == steps and row["init_mode"] == mode and row["method"] == method
                ]
                vals.append(fmt(float(np.median(ratios))) if ratios else "--")
            lines.append(f"{steps} & {labels[mode]} & " + " & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--init-out", type=Path, default=None)
    args = parser.parse_args()
    rows = load_jsonl(args.jsonl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(make_table(rows), encoding="utf-8")
    if args.init_out is not None:
        args.init_out.parent.mkdir(parents=True, exist_ok=True)
        args.init_out.write_text(make_init_table(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
