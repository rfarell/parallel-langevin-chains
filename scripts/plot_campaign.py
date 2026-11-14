import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["iid", "zero_sum", "antithetic", "common"]
LABELS = {
    "iid": "iid",
    "zero_sum": "centered ($C_\\star$)",
    "antithetic": "pairwise antithetic",
    "common": "common",
}
# Table wording is fixed separately so figure-legend tweaks never alter .tex output.
TABLE_LABELS = {
    "iid": "iid",
    "zero_sum": "$C_\\star$",
    "antithetic": "antithetic",
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


def panel_name(row):
    if row.get("target") == "real_logistic":
        return row["dataset"]
    return row["target"]


def metric_name(row):
    return "mse" if row.get("target") == "real_logistic" else "lin_ens_var"


def fmt(x):
    if x is None or not np.isfinite(x):
        return "--"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.2e}"
    return f"{x:.3f}"


def fmt_k(x):
    if x is None or not np.isfinite(x):
        return "--"
    if x > 1e6:
        return "$>10^6$"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def tex_label(name):
    return name.replace("_", " ")


def aggregate(rows):
    grouped = {}
    for row in rows:
        key = (panel_name(row), int(row["chains"]), row.get("method", row.get("coupling")))
        metric = metric_name(row)
        grouped.setdefault(key, {"metric": [], "elapsed": [], "rows": []})
        grouped[key]["metric"].append(float(row[metric]))
        grouped[key]["elapsed"].append(float(row.get("elapsed_sec_per_ensemble", row["elapsed_sec"] / row.get("reps", row.get("n_ensembles", 1)))))
        grouped[key]["rows"].append(row)
    out = {}
    for key, values in grouped.items():
        out[key] = {
            "metric": float(np.median(values["metric"])),
            "metric_q25": float(np.quantile(values["metric"], 0.25)),
            "metric_q75": float(np.quantile(values["metric"], 0.75)),
            "elapsed": float(np.median(values["elapsed"])),
            "rows": values["rows"],
        }
    return out


def equivalent_iid_k(agg, panel, k, method):
    value = agg.get((panel, k, method), {}).get("metric")
    iid_pairs = [(kk, val["metric"]) for (pp, kk, mm), val in agg.items() if pp == panel and mm == "iid" and val["metric"] > 0]
    if value is None or value <= 0 or not iid_pairs:
        return math.inf
    constants = [kk * metric for kk, metric in iid_pairs]
    c = float(np.median(constants))
    return c / value


def real_iid_lookup(rows):
    out = {}
    for row in rows:
        if row.get("target") != "real_logistic":
            continue
        if row.get("method") != "iid":
            continue
        key = (row["dataset"], int(row["chains"]), row.get("prior"), row.get("replicate"))
        out[key] = row
    return out


def paired_real_ratios(rows, dataset, chains, method, metric):
    iid = real_iid_lookup(rows)
    values = []
    for row in rows:
        if row.get("target") != "real_logistic":
            continue
        if row["dataset"] != dataset or int(row["chains"]) != chains or row.get("method") != method:
            continue
        ref = iid.get((dataset, chains, row.get("prior"), row.get("replicate")))
        if ref is None or float(ref.get(metric, 0.0)) <= 0:
            continue
        values.append(float(row[metric]) / float(ref[metric]))
    return values


def median_iqr(values):
    if not values:
        return None, None, None
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def fmt_interval(values):
    med, q25, q75 = median_iqr(values)
    if med is None:
        return "--"
    return f"{fmt(med)} [{fmt(q25)}, {fmt(q75)}]"


def plot_wallclock(rows, out_path):
    agg = aggregate(rows)
    panels = sorted({panel_name(row) for row in rows}, key=lambda x: ["gaussian", "quartic", "synthetic_logistic", "wdbc", "spambase", "adult"].index(x) if x in ["gaussian", "quartic", "synthetic_logistic", "wdbc", "spambase", "adult"] else 99)
    ncols = 3
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        for method in METHODS:
            points = sorted((k, values["elapsed"], values["metric"]) for (pp, k, mm), values in agg.items() if pp == panel and mm == method)
            if not points:
                continue
            ks, xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=LABELS[method], color=COLORS[method])
            for k, x, y in points:
                if k in {8, 64, 128, 256}:
                    ax.annotate(str(k), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
        zero = agg.get((panel, 8, "zero_sum"))
        if zero:
            eq = equivalent_iid_k(agg, panel, 8, "zero_sum")
            ax.axhline(zero["metric"], color=COLORS["zero_sum"], linestyle="--", linewidth=0.9, alpha=0.7)
            label = "metric-equiv iid K≈" + fmt_k(eq)
            ax.text(0.02, 0.05, label, transform=ax.transAxes, fontsize=8, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(panel.replace("_", " "))
        ax.set_xlabel("seconds per ensemble")
        ax.set_ylabel("MSE" if panel in {"wdbc", "spambase", "adult"} else r"Var$(a^\top \bar X_t)$")
        ax.grid(True, which="both", alpha=0.25)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def real_summary_table(rows, out_path):
    panels = [p for p in ["wdbc", "spambase", "adult"] if any(panel_name(row) == p for row in rows)]
    lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Dataset & coupling & MSE/iid [IQR] & var/iid & bias/MSE & log loss & sec/ens \\\\",
        "\\midrule",
    ]
    for panel in panels:
        for method in METHODS:
            method_rows = [
                r for r in rows
                if r.get("target") == "real_logistic"
                and r["dataset"] == panel
                and int(r["chains"]) == 8
                and r.get("method") == method
            ]
            if not method_rows:
                continue
            mse_ratio = paired_real_ratios(rows, panel, 8, method, "mse")
            var_ratio = paired_real_ratios(rows, panel, 8, method, "trace_var")
            bias_share = [
                float(r.get("bias_sq", np.nan)) / float(r["mse"])
                for r in method_rows
                if float(r.get("mse", 0.0)) > 0
            ]
            logloss = [float(r.get("test_logloss", np.nan)) for r in method_rows]
            elapsed = [float(r.get("elapsed_sec_per_ensemble", np.nan)) for r in method_rows]
            name = TABLE_LABELS[method]
            lines.append(
                f"{panel} & {name} & {fmt_interval(mse_ratio)} & "
                f"{fmt(median_iqr(var_ratio)[0])} & {fmt(median_iqr(bias_share)[0])} & "
                f"{fmt(median_iqr(logloss)[0])} & {fmt(median_iqr(elapsed)[0])} \\\\"
            )
        lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_real_k_sweep(rows, out_path):
    datasets = [p for p in ["wdbc", "spambase", "adult"] if any(panel_name(row) == p for row in rows)]
    if not datasets:
        return
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.1 * len(datasets), 6.0), squeeze=False, sharex="col")
    metrics = [("mse", "MSE / iid"), ("trace_var", "trace var / iid")]
    for col, dataset in enumerate(datasets):
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col]
            for method in METHODS:
                xs, meds, lows, highs = [], [], [], []
                ks = sorted({
                    int(r["chains"]) for r in rows
                    if r.get("target") == "real_logistic" and r["dataset"] == dataset and r.get("method") == method
                })
                for k in ks:
                    vals = paired_real_ratios(rows, dataset, k, method, metric)
                    med, q25, q75 = median_iqr(vals)
                    if med is None:
                        continue
                    xs.append(k)
                    meds.append(med)
                    lows.append(q25)
                    highs.append(q75)
                if not xs:
                    continue
                ax.plot(xs, meds, marker="o", label=LABELS[method], color=COLORS[method])
                if len(xs) > 1:
                    ax.fill_between(xs, lows, highs, color=COLORS[method], alpha=0.13, linewidth=0)
            ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(sorted({int(r["chains"]) for r in rows if r.get("target") == "real_logistic" and r["dataset"] == dataset}))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.grid(True, which="both", alpha=0.25)
            if row_idx == 0:
                ax.set_title(dataset)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 1:
                ax.set_xlabel("chains")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def wallclock_table(rows, out_path):
    agg = aggregate(rows)
    preferred = ["gaussian", "quartic", "synthetic_logistic", "wdbc", "spambase", "adult"]
    panels = [p for p in preferred if any(panel_name(row) == p for row in rows)]
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Target & $C_\\star$ K=8 metric & metric-equiv. iid K & $C_\\star$ sec/ens & overhead \\\\",
        "\\midrule",
    ]
    for panel in panels:
        zero = agg.get((panel, 8, "zero_sum"))
        iid = agg.get((panel, 8, "iid"))
        if not zero or not iid:
            continue
        eq = equivalent_iid_k(agg, panel, 8, "zero_sum")
        overhead = zero["elapsed"] / iid["elapsed"] if iid["elapsed"] > 0 else np.nan
        lines.append(f"{tex_label(panel)} & {fmt(zero['metric'])} & {fmt_k(eq)} & {fmt(zero['elapsed'])} & {fmt(overhead)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--fig", type=Path, required=True)
    parser.add_argument("--real-table", type=Path, required=True)
    parser.add_argument("--wallclock-table", type=Path, required=True)
    parser.add_argument("--real-k-fig", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.jsonl)
    plot_wallclock(rows, args.fig)
    plot_real_k_sweep(rows, args.real_k_fig)
    real_summary_table(rows, args.real_table)
    wallclock_table(rows, args.wallclock_table)


if __name__ == "__main__":
    main()
