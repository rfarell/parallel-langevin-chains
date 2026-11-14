import argparse
import json
from pathlib import Path

import numpy as np

DATASET_LABELS = {
    "wdbc": "WDBC",
    "spambase": "Spambase",
    "adult": "Adult",
    "mnist79": "MNIST 7v9",
    "gisette": "Gisette",
}


def load_jsonl(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def load_campaign(sweeps_dir, stem, reps=(0, 1, 2)):
    return load_jsonl([sweeps_dir / f"{stem}_rep{r}.jsonl" for r in reps])


def latex_sci(x, sig=3):
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    return f"${mant}\\times 10^{{{int(exp)}}}$"


def fmt(x, sig=3):
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e3:
        return latex_sci(x, sig)
    decimals = max(sig - 1 - int(np.floor(np.log10(abs(x)))), 0)
    return f"{x:.{decimals}f}"


def cell(vals, sig=3):
    med = float(np.median(vals))
    return f"{fmt(med, sig)} [{fmt(min(vals), sig)}, {fmt(max(vals), sig)}]"


def ratios(rows, key, num_method, field="trace_var", den_method="iid"):
    """Per-replicate num/den ratios for rows matching key (dict of field: value)."""
    sub = [r for r in rows if all(r.get(k) == v for k, v in key.items())]
    out = []
    for rep in sorted({r["replicate"] for r in sub}):
        num = [r[field] for r in sub if r["replicate"] == rep and r["method"] == num_method]
        den = [r[field] for r in sub if r["replicate"] == rep and r["method"] == den_method]
        assert len(num) == 1 and len(den) == 1, f"expected 1 row/method, got {key} rep{rep}: {len(num)}/{len(den)}"
        out.append(num[0] / den[0])
    assert len(out) == 3, f"expected 3 replicates for {key}, got {len(out)}"
    return out


def absolutes(rows, key, method, field):
    sub = [r[field] for r in rows if all(r.get(k) == v for k, v in key.items()) and r["method"] == method]
    assert len(sub) == 3, f"expected 3 replicates for {key} {method} {field}, got {len(sub)}"
    return sub


def dataset_prior_cells(rows, datasets, priors):
    for ds in datasets:
        for prior in priors:
            yield ds, prior


def make_mala_table(rows, summary):
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & prior & ULA $C_\\star$/iid & MALA $C_\\star$/iid & MALA pairwise/iid & MALA accept \\\\",
        "\\midrule",
    ]
    summary.append("== mala_real.tex (trace-var ratios, median across 3 reps) ==")
    for ds, prior in dataset_prior_cells(rows, ["wdbc", "spambase", "adult"], [0.1, 0.3]):
        ula = ratios(rows, {"dataset": ds, "prior": prior, "kernel": "ula"}, "zero_sum")
        mala = ratios(rows, {"dataset": ds, "prior": prior, "kernel": "mala"}, "zero_sum")
        mala_pair = ratios(rows, {"dataset": ds, "prior": prior, "kernel": "mala"}, "antithetic")
        accept = absolutes(rows, {"dataset": ds, "prior": prior, "kernel": "mala"}, "zero_sum", "accept_rate")
        acc = f"{float(np.median(accept)):.2f}"
        lines.append(
            f"{DATASET_LABELS[ds]} & {prior} & {cell(ula)} & {cell(mala)} & {cell(mala_pair)} & {acc} \\\\"
        )
        summary.append(
            f"  {ds} prior={prior}: ULA C*/iid={np.median(ula):.3g} MALA C*/iid={np.median(mala):.3g} "
            f"MALA pair/iid={np.median(mala_pair):.3g} accept={acc}"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_uncertainty_table(rows, summary):
    lines = [
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Dataset & prior & $\\hat{v}$ $C_\\star$/iid & $\\hat{v}$ pairwise/iid & CIW $C_\\star$/iid "
        "& CIW pairwise/iid & pred.\\ $C_\\star$/iid & pred.\\ pairwise/iid \\\\",
        "\\midrule",
    ]
    summary.append("== uncertainty_observables.tex (observable-variance ratios) ==")
    for ds, prior in dataset_prior_cells(rows, ["wdbc", "spambase", "adult"], [0.1, 0.3]):
        key = {"dataset": ds, "prior": prior}
        vhat_zs = ratios(rows, key, "zero_sum", "vhat_var_sum")
        vhat_pa = ratios(rows, key, "antithetic", "vhat_var_sum")
        ciw_zs = ratios(rows, key, "zero_sum", "ciw_var_sum")
        ciw_pa = ratios(rows, key, "antithetic", "ciw_var_sum")
        pred_zs = ratios(rows, key, "zero_sum", "pred_var_sum")
        pred_pa = ratios(rows, key, "antithetic", "pred_var_sum")
        cells = [cell(v) for v in (vhat_zs, vhat_pa, ciw_zs, ciw_pa, pred_zs, pred_pa)]
        lines.append(f"{DATASET_LABELS[ds]} & {prior} & " + " & ".join(cells) + " \\\\")
        summary.append(
            f"  {ds} prior={prior}: vhat C*={np.median(vhat_zs):.3g} pair={np.median(vhat_pa):.3g} "
            f"CIW C*={np.median(ciw_zs):.3g} pair={np.median(ciw_pa):.3g} "
            f"pred C*={np.median(pred_zs):.3g} pred pair={np.median(pred_pa):.3g}"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_nonlogconcave_table(rows, summary):
    tp = [r for r in rows if r["model"] == "tprior"]
    hi = [r for r in rows if r["model"] == "hierarchical"]
    df = tp[0]["df"]
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        f"\\multicolumn{{6}}{{l}}{{\\emph{{Student-$t$ prior ($\\nu={df:g}$)}}}} \\\\",
        "Dataset & prior & $C_\\star$/iid & pairwise/iid & & \\\\",
        "\\midrule",
    ]
    summary.append("== nonlogconcave_real.tex (trace-var ratios) ==")
    for ds, prior in dataset_prior_cells(tp, ["wdbc", "spambase", "adult"], [0.1, 0.3]):
        key = {"dataset": ds, "prior": prior}
        zs = ratios(tp, key, "zero_sum")
        pa = ratios(tp, key, "antithetic")
        lines.append(f"{DATASET_LABELS[ds]} & {prior} & {cell(zs)} & {cell(pa)} & & \\\\")
        summary.append(f"  tprior {ds} prior={prior}: C*/iid={np.median(zs):.3g} pair/iid={np.median(pa):.3g}")
    key = {"dataset": "adult"}
    zs_full = ratios(hi, key, "zero_sum", "trace_var")
    zs_beta = ratios(hi, key, "zero_sum", "trace_var_beta")
    pa_full = ratios(hi, key, "antithetic", "trace_var")
    pa_beta = ratios(hi, key, "antithetic", "trace_var_beta")
    prior = hi[0]["prior"]
    lines.extend([
        "\\midrule",
        "\\multicolumn{6}{l}{\\emph{Hierarchical logistic (Adult)}} \\\\",
        "Dataset & prior & full $C_\\star$/iid & $\\beta$-block $C_\\star$/iid "
        "& full pairwise/iid & $\\beta$-block pairwise/iid \\\\",
        "\\midrule",
        f"Adult & {prior} & {cell(zs_full)} & {cell(zs_beta)} & {cell(pa_full)} & {cell(pa_beta)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])
    summary.append(
        f"  hierarchical adult prior={prior}: full C*/iid={np.median(zs_full):.3g} "
        f"beta-block C*/iid={np.median(zs_beta):.3g} full pair/iid={np.median(pa_full):.3g} "
        f"beta-block pair/iid={np.median(pa_beta):.3g}"
    )
    return "\n".join(lines)


def make_highdim_table(rows, summary):
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & prior & $d$ & $C_\\star$/iid & pairwise/iid & common/iid \\\\",
        "\\midrule",
    ]
    summary.append("== highdim_real.tex (trace-var ratios) ==")
    for ds, prior in dataset_prior_cells(rows, ["mnist79", "gisette"], [0.1, 0.3]):
        key = {"dataset": ds, "prior": prior}
        dim = {r["dim"] for r in rows if r["dataset"] == ds}
        assert len(dim) == 1
        zs = ratios(rows, key, "zero_sum")
        pa = ratios(rows, key, "antithetic")
        co = ratios(rows, key, "common")
        lines.append(
            f"{DATASET_LABELS[ds]} & {prior} & {dim.pop()} & {cell(zs)} & {cell(pa)} & {cell(co)} \\\\"
        )
        summary.append(
            f"  {ds} prior={prior}: C*/iid={np.median(zs):.3g} pair/iid={np.median(pa):.3g} "
            f"common/iid={np.median(co):.3g}"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_bayes_linreg_table(rows, summary):
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "$\\lambda$ & dtype & iid trace var & $C_\\star$ trace var & $C_\\star$/iid "
        "& $C_\\star$ MSE & $C_\\star$ bias$^2$ \\\\",
        "\\midrule",
    ]
    summary.append("== bayes_linreg_exact.tex (absolute trace vars; MSE vs closed-form mean) ==")
    dtypes = sorted({r["dtype"] for r in rows})
    for lam in [0.1, 0.3]:
        for dtype in dtypes:
            key = {"lam": lam, "dtype": dtype}
            iid_tv = absolutes(rows, key, "iid", "trace_var")
            zs_tv = absolutes(rows, key, "zero_sum", "trace_var")
            ratio = ratios(rows, key, "zero_sum")
            zs_mse = absolutes(rows, key, "zero_sum", "mse")
            zs_bias = absolutes(rows, key, "zero_sum", "bias_sq")
            lines.append(
                f"{lam} & {dtype} & {cell(iid_tv)} & {cell(zs_tv)} & {cell(ratio)} "
                f"& {cell(zs_mse)} & {cell(zs_bias)} \\\\"
            )
            summary.append(
                f"  lam={lam} {dtype}: iid var={np.median(iid_tv):.3g} C* var={np.median(zs_tv):.3g} "
                f"ratio={np.median(ratio):.3g} C* MSE={np.median(zs_mse):.3g} C* bias^2={np.median(zs_bias):.3g}"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize new JSONL campaigns into LaTeX table fragments.")
    parser.add_argument("--sweeps-dir", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--tables-dir", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    jobs = [
        ("mala_real", "mala_real.tex", make_mala_table),
        ("uncertainty_obs", "uncertainty_observables.tex", make_uncertainty_table),
        ("nonlogconcave", "nonlogconcave_real.tex", make_nonlogconcave_table),
        ("highdim_real", "highdim_real.tex", make_highdim_table),
        ("bayes_linreg", "bayes_linreg_exact.tex", make_bayes_linreg_table),
    ]
    for stem, out_name, builder in jobs:
        rows = load_campaign(args.sweeps_dir, stem)
        out_path = args.tables_dir / out_name
        out_path.write_text(builder(rows, summary), encoding="utf-8")
        summary.append(f"  -> wrote {out_path}")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
