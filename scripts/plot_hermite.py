import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter


LABELS = {
    "iid": "iid",
    "zero_sum": "centered ($C_\\star$)",
    "antithetic": "pairwise antithetic",
    "common": "common",
}
COLORS = {
    "iid": "#1f77b4",
    "zero_sum": "#2ca02c",
    "antithetic": "#ff7f0e",
    "common": "#d62728",
}


def load_jsonl(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def median_by(rows, method):
    out = {}
    theory = {}
    for row in rows:
        if row["method"] != method:
            continue
        key = row["beta"]
        out.setdefault(key, []).append(max(row["empirical_ratio_to_chain"], 1e-15))
        theory.setdefault(key, []).append(max(row["theory_ratio_to_chain"], 1e-15))
    xs = sorted(out)
    return xs, [np.median(out[x]) for x in xs], [np.median(theory[x]) for x in xs]


def plot(rows, out_path):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    all_xs = set()
    for method in ["iid", "zero_sum", "antithetic"]:
        xs, ys, th = median_by(rows, method)
        if not xs:
            continue
        all_xs.update(xs)
        ax.plot(xs, ys, marker="o", markersize=4.5, color=COLORS[method], label=f"{LABELS[method]} empirical")
        ax.plot(xs, th, linestyle="--", linewidth=1.2, color=COLORS[method], alpha=0.8, label=f"{LABELS[method]} theory")
    ax.set_xscale("symlog", linthresh=0.2)
    ax.set_yscale("log")
    ticks = sorted(all_xs)
    if ticks and len(ticks) <= 9:
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"quadratic/linear amplitude $\beta/\alpha$", fontsize=11)
    ax.set_ylabel(r"Var(mean $f$) / Var(single chain)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fmt(x):
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.3f}"


def table(rows, out_path):
    methods = ["iid", "zero_sum", "antithetic"]
    labels = {"iid": "iid", "zero_sum": "$C_\\star$", "antithetic": "antithetic"}
    lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Method & median abs. error & max abs. error \\\\",
        "\\midrule",
    ]
    for method in methods:
        vals = [row["abs_error"] for row in rows if row["method"] == method]
        lines.append(f"{labels[method]} & {fmt(float(np.median(vals)))} & {fmt(float(np.max(vals)))} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--fig", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.jsonl)
    plot(rows, args.fig)
    table(rows, args.table)


if __name__ == "__main__":
    main()
