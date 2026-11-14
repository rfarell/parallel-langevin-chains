import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter


LABELS = {
    "iid": "iid",
    "zero_sum": "zero-sum ($C_\\star$)",
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


def medians(rows, target):
    out = {}
    for row in rows:
        if row["target"] != target:
            continue
        key = (row["chains"], row["method"])
        out.setdefault(key, []).append(max(row["empirical_ratio_to_chain"], 1e-15))
    ks = sorted({k for k, _ in out})
    methods = ["iid", "zero_sum", "antithetic", "common"]
    return ks, methods, {
        method: [np.median(out.get((k, method), [np.nan])) for k in ks]
        for method in methods
    }


def plot(rows, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0), sharey=True)
    for ax, target in zip(axes, ["gaussian", "quartic", "logistic"]):
        ks, methods, data = medians(rows, target)
        if not ks:
            ax.set_visible(False)
            continue
        for method in methods:
            ax.plot(ks, data[method], marker="o", markersize=4.5, label=LABELS[method], color=COLORS[method])
        if target == "gaussian":
            ax.plot(ks, [1 / (k * k) for k in ks], color="black", linestyle=":", label="odd antithetic theory")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if len(ks) <= 9:
            ax.set_xticks(list(ks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(target, fontsize=11)
        ax.set_xlabel("chains", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("variance ratio", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def fmt(x):
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.3f}"


def table(rows, out_path):
    targets = ["gaussian", "quartic", "logistic"]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Target & iid & zero-sum & odd antithetic & common \\\\",
        "\\midrule",
    ]
    for target in targets:
        vals = []
        for method in ["iid", "zero_sum", "antithetic", "common"]:
            ratios = [
                max(row["empirical_ratio_to_chain"], 1e-15)
                for row in rows
                if row["target"] == target and row["method"] == method
            ]
            vals.append(fmt(float(np.median(ratios))) if ratios else "--")
        lines.append(f"{target} & " + " & ".join(vals) + " \\\\")
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
