"""Emit tables for the MALA rejection-leak law and the exact-gold end-task check."""
import argparse
import collections
import json
import statistics
from pathlib import Path

DATASET_LABEL = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows


def med_range(vals, fmt):
    return f"{fmt.format(statistics.median(vals))} [{fmt.format(min(vals))}, {fmt.format(max(vals))}]"


def mala_law_table(rows, out):
    cells = collections.defaultdict(dict)
    for r in rows:
        cells[(r["dataset"], r["prior"], r["replicate"])][r["method"]] = r
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for (ds, prior, _), m in cells.items():
        if "zero_sum" not in m or "iid" not in m:
            continue
        z = m["zero_sum"]
        p = 1.0 - z["accept_rate"]
        s = z["noise_sq_given_reject"] / z["noise_sq_overall"]
        agg[(ds, prior)]["p"].append(p)
        agg[(ds, prior)]["s"].append(s)
        agg[(ds, prior)]["ps"].append(p * s)
        agg[(ds, prior)]["vr"].append(z["trace_var"] / m["iid"]["trace_var"])
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & prior & rejection rate $p$ & selection $s$ & predicted $p\\,s$ & measured $C_\\star$/iid \\\\",
        "\\midrule",
    ]
    for ds in ["wdbc", "spambase", "adult"]:
        for prior in [0.1, 0.3]:
            a = agg[(ds, prior)]
            lines.append(
                f"{DATASET_LABEL[ds]} & {prior} & {statistics.median(a['p']):.3f} & "
                f"{statistics.median(a['s']):.3f} & {statistics.median(a['ps']):.3f} & "
                f"{med_range(a['vr'], '{:.3f}')} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def gold_table(rows, out, map_rows=None, vi_rows=None):
    est = [r for r in rows if r.get("kind") == "estimator" and r.get("pair") in ("primary", "context")]
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in est:
        key = (r["dataset"], r["prior"])
        if r["estimator"] == "single" and r.get("pair") == "primary":
            name = f"{r['coupling']}@{'e1' if r['level'] == 'eta1' else 'e2'}"
        elif r["estimator"] == "mala":
            name = "mala"
        else:
            continue
        cells[key][name].append(r["mse_gold_median"])
    golds = {}
    for r in rows:
        if r.get("kind") == "gold":
            golds[(r["dataset"], r["prior"])] = r["gold_spread"]
    maps = {}
    for r in map_rows or []:
        maps[(r["dataset"], r["prior"])] = r["map_err_sq_gold"]
    vis, eps_ = {}, {}
    for r in vi_rows or []:
        if r.get("kind") == "vi_baseline":
            vis[(r["dataset"], r["prior"])] = r["vi_err_sq_gold"]
        elif r.get("kind") == "ep_baseline" and r.get("converged"):
            eps_[(r["dataset"], r["prior"])] = r["ep_err_sq_gold"]
    all_cells = [(ds, prior) for ds in ["wdbc", "spambase", "adult"] for prior in [0.1, 0.3]]
    # Pre-registered all-or-none rule: the EP column appears only if EP
    # converged on every cell; partial results stay in the JSONL artifact.
    show_ep = bool(eps_) and all(c in eps_ for c in all_cells)
    show_vi = bool(vis) and all(c in vis for c in all_cells)
    header = "Dataset & prior & MAP"
    colspec = "llr"
    if show_vi:
        header += " & VI"
        colspec += "r"
    if show_ep:
        header += " & EP"
        colspec += "r"
    header += " & iid ULA & $C_\\star$ ULA & $C_\\star$ ULA (fine) & MALA & gold spread \\\\"
    lines = [
        f"\\begin{{tabular}}{{{colspec}rrrrr}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for ds in ["wdbc", "spambase", "adult"]:
        for prior in [0.1, 0.3]:
            c = cells[(ds, prior)]
            map_cell = f"{maps[(ds, prior)]:.3g}" if (ds, prior) in maps else "---"
            parts = [f"{DATASET_LABEL[ds]} & {prior}", map_cell]
            if show_vi:
                parts.append(f"{vis[(ds, prior)]:.3g}")
            if show_ep:
                parts.append(f"{eps_[(ds, prior)]:.3g}")
            parts += [
                med_range(c["iid@e1"], "{:.3g}"),
                med_range(c["zero_sum@e1"], "{:.3g}"),
                med_range(c["zero_sum@e2"], "{:.3g}"),
                med_range(c["mala"], "{:.3g}"),
                f"{golds[(ds, prior)]:.1e}",
            ]
            lines.append(" & ".join(parts) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"wrote {out} (vi column: {show_vi}, ep column: {show_ep})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", nargs="+", default=[f"outputs/sweeps/mala_diag_rep{r}.jsonl" for r in range(3)])
    parser.add_argument("--rr", nargs="+", default=[f"outputs/sweeps/rr_extrapolation_rep{r}.jsonl" for r in range(3)])
    parser.add_argument("--map", default="outputs/sweeps/map_baseline.jsonl")
    parser.add_argument("--vi", default="outputs/sweeps/vi_baseline.jsonl")
    parser.add_argument("--law-out", default="paper/tables/mala_law.tex")
    parser.add_argument("--gold-out", default="paper/tables/gold_endtask.tex")
    args = parser.parse_args()
    mala_law_table(load(args.diag), args.law_out)
    map_rows = load([args.map]) if Path(args.map).exists() else None
    vi_rows = load([args.vi]) if Path(args.vi).exists() else None
    gold_table(load(args.rr), args.gold_out, map_rows=map_rows, vi_rows=vi_rows)


if __name__ == "__main__":
    main()
