import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter


METHODS = ["iid", "zero_sum", "antithetic", "common"]
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
INIT_MODES = ["iid_random", "zero_sum_random", "common_random", "common_zero"]
INIT_LABELS = {
    "iid_random": "iid random",
    "zero_sum_random": "zero-sum random",
    "common_random": "common random",
    "common_zero": "common zero",
}
INIT_COLORS = {
    "iid_random": "#1f77b4",
    "zero_sum_random": "#2ca02c",
    "common_random": "#d62728",
    "common_zero": "#7f7f7f",
}
XLABELS = {
    "chains": "chains",
    "dim": "dimension $d$",
    "beta": r"quartic strength $\beta$",
}


def ordered_groups(groups, preferred):
    return [g for g in preferred if g in groups] + [g for g in groups if g not in preferred]


def plain_log_xticks(ax, xs, max_ticks=9):
    if len(xs) <= max_ticks:
        ax.set_xticks(list(xs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def load_jsonl(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def median_by(rows, target, x_key, group_key="method", metric="empirical_ratio_to_chain", floor=1e-15):
    out = {}
    for row in rows:
        if row["target"] != target:
            continue
        if "init_mode" in row:
            continue
        if metric not in row:
            continue
        key = (row[x_key], row[group_key])
        out.setdefault(key, []).append(max(row[metric], floor))
    xs = sorted({k[0] for k in out})
    groups = sorted({k[1] for k in out})
    return xs, groups, {g: [np.median(out.get((x, g), [np.nan])) for x in xs] for g in groups}


def median_by_filter(rows, keep, x_key, group_key, metric="empirical_ratio_to_chain", floor=1e-15):
    out = {}
    for row in rows:
        if not keep(row):
            continue
        if metric not in row:
            continue
        key = (row[x_key], row[group_key])
        out.setdefault(key, []).append(max(row[metric], floor))
    xs = sorted({k[0] for k in out})
    groups = sorted({k[1] for k in out})
    return xs, groups, {g: [np.median(out.get((x, g), [np.nan])) for x in xs] for g in groups}


def plot_ratio(
    rows,
    target,
    x_key,
    out_path,
    theory_iid=False,
    metric="empirical_ratio_to_chain",
    ylabel=None,
):
    xs, groups, data = median_by(rows, target, x_key, metric=metric)
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for group in ordered_groups(groups, METHODS):
        ax.plot(
            xs,
            data[group],
            marker="o",
            markersize=4.5,
            label=LABELS.get(group, group),
            color=COLORS.get(group),
        )
    if theory_iid:
        theory = [1.0 / x for x in xs]
        ax.plot(xs, theory, color="black", linestyle="--", linewidth=1.2, label="iid theory")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_xticks(ax, xs)
    ax.set_xlabel(XLABELS.get(x_key, x_key), fontsize=11)
    ax.set_ylabel(ylabel or r"Var$(a^\top \bar X_t)$ / Var$(a^\top X_t^{(1)})$", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", alpha=0.25)
    # Keep legends clear of the centered curves in the archived paper panels.
    legend_position = {}
    if target == "quartic":
        legend_position = {"loc": "lower right", "bbox_to_anchor": (1, 0.075)}
    elif target == "logistic":
        legend_position = {"loc": "center left", "bbox_to_anchor": (0.01, 0.61)}
    ax.legend(frameon=True, framealpha=1.0, edgecolor="none",
              fontsize=7 if target == "quartic" else 8, **legend_position)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_init_horizon(rows, out_path):
    xs, groups, data = median_by_filter(
        rows,
        keep=lambda row: row.get("target") == "gaussian" and row.get("method") == "zero_sum" and "init_mode" in row,
        x_key="steps",
        group_key="init_mode",
    )
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for group in ordered_groups(groups, INIT_MODES):
        ax.plot(
            xs,
            data[group],
            marker="o",
            markersize=4.5,
            label=INIT_LABELS.get(group, group),
            color=INIT_COLORS.get(group),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_xticks(ax, xs)
    ax.set_xlabel("Langevin steps", fontsize=11)
    ax.set_ylabel(r"Var$(a^\top \bar X_t)$ / Var$(a^\top X_t^{(1)})$", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, framealpha=1.0, edgecolor="none", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_init_methods(rows, out_path):
    xs, groups, data = median_by_filter(
        rows,
        keep=lambda row: row.get("target") == "gaussian" and row.get("init_mode") == "iid_random",
        x_key="steps",
        group_key="method",
    )
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for group in ordered_groups(groups, METHODS):
        ax.plot(
            xs,
            data[group],
            marker="o",
            markersize=4.5,
            label=LABELS.get(group, group),
            color=COLORS.get(group),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_xticks(ax, xs)
    ax.set_xlabel("Langevin steps", fontsize=11)
    ax.set_ylabel(r"Var$(a^\top \bar X_t)$ / Var$(a^\top X_t^{(1)})$", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=True, framealpha=1.0, edgecolor="none", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    rows = load_jsonl(args.jsonl)
    plot_ratio(rows, "gaussian", "chains", args.outdir / "gaussian_chain_ratio.png", theory_iid=True)
    plot_ratio(rows, "gaussian", "dim", args.outdir / "gaussian_dim_ratio.png")
    plot_ratio(
        rows,
        "gaussian",
        "chains",
        args.outdir / "gaussian_quadratic_observable_ratio.png",
        theory_iid=True,
        metric="quad_empirical_ratio_to_chain",
        ylabel="quadratic observable ratio",
    )
    plot_ratio(rows, "quartic", "beta", args.outdir / "quartic_beta_ratio.png")
    plot_ratio(rows, "logistic", "chains", args.outdir / "logistic_chain_ratio.png", theory_iid=True)
    plot_ratio(rows, "logistic", "dim", args.outdir / "logistic_dim_ratio.png")
    plot_init_horizon(rows, args.outdir / "gaussian_init_horizon_ratio.png")
    plot_init_methods(rows, args.outdir / "gaussian_iid_init_method_ratio.png")


if __name__ == "__main__":
    main()
