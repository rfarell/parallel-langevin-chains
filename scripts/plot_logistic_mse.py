import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def design_key(row):
    return (row["dim"], row["chains"], row["prior"], row["n_data"], row["design_seed"], row["seed"])


def ratios_to_iid(rows, metric):
    by_design = {}
    for row in rows:
        by_design.setdefault(design_key(row), {})[row["method"]] = row
    out = {method: [] for method in ["iid", "zero_sum", "antithetic", "common"]}
    for methods in by_design.values():
        if "iid" not in methods:
            continue
        denom = methods["iid"][metric]
        if denom <= 0:
            continue
        for method, row in methods.items():
            out[method].append(row[metric] / denom)
    return out


def fmt(x):
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return f"{x:.2e}"
    return f"{x:.3f}"


def table(rows, out_path):
    labels = {
        "iid": "iid",
        "zero_sum": "zero-sum",
        "antithetic": "antithetic",
        "common": "common",
    }
    mse = ratios_to_iid(rows, "mse")
    var = ratios_to_iid(rows, "trace_var")
    single = {m: [row["mse_to_single"] for row in rows if row["method"] == m] for m in labels}
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & med MSE/iid & q90 & max & med var/iid & med MSE/single \\\\",
        "\\midrule",
    ]
    for method in ["iid", "zero_sum", "antithetic", "common"]:
        method_mse = np.asarray(mse[method], dtype=float)
        lines.append(
            f"{labels[method]} & {fmt(float(np.median(method_mse)))} & "
            f"{fmt(float(np.quantile(method_mse, 0.9)))} & "
            f"{fmt(float(np.max(method_mse)))} & "
            f"{fmt(float(np.median(var[method])))} & "
            f"{fmt(float(np.median(single[method])))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


PLOT_LABELS = {
    "iid": "iid",
    "zero_sum": "zero-sum\n($C_\\star$)",
    "antithetic": "pairwise\nantithetic",
    "common": "common",
}
COLORS = {
    "iid": "#1f77b4",
    "zero_sum": "#2ca02c",
    "antithetic": "#ff7f0e",
    "common": "#d62728",
}


def plot(rows, out_path):
    ratios = ratios_to_iid(rows, "mse")
    methods = ["iid", "zero_sum", "antithetic", "common"]
    data = [ratios[m] for m in methods]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    box = ax.boxplot(
        data,
        tick_labels=[PLOT_LABELS[m] for m in methods],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(mcolors.to_rgba(COLORS[method], 0.45))
        patch.set_edgecolor(COLORS[method])
    for i, method in enumerate(methods):
        for part in ("whiskers", "caps"):
            for artist in box[part][2 * i:2 * i + 2]:
                artist.set_color(COLORS[method])
    ax.set_yscale("log")
    ax.set_ylabel("posterior-mean MSE / iid MSE", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


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
