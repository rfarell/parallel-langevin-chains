"""Emit paper/tables/oddk_zerosum_class.tex from oddk_zerosum_rep{0,1,2}.jsonl.

Columns are trace-var ratios vs iid (median [min, max] across replicates),
matching the cell/fmt conventions of summarize_new_experiments.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

DATASET_LABELS = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}
COLUMNS = [("zero_sum", "$C_\\star$/iid"),
           ("pairs_plus_triplet", "pairs+triplet/iid"),
           ("pair_plus_one", "pair+1/iid")]


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
    sub = [r for r in rows if all(r.get(k) == v for k, v in key.items())]
    out = []
    for rep in sorted({r["replicate"] for r in sub}):
        num = [r[field] for r in sub if r["replicate"] == rep and r["method"] == num_method]
        den = [r[field] for r in sub if r["replicate"] == rep and r["method"] == den_method]
        assert len(num) == 1 and len(den) == 1, f"expected 1 row/method, got {key} rep{rep}: {len(num)}/{len(den)}"
        out.append(num[0] / den[0])
    assert len(out) == 3, f"expected 3 replicates for {key}, got {len(out)}"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--pattern", default="oddk_zerosum_rep{rep}.jsonl")
    parser.add_argument("--out", type=Path, default=Path("paper/tables/oddk_zerosum_class.tex"))
    args = parser.parse_args()
    rows = []
    for rep in (0, 1, 2):
        path = args.sweep_dir / args.pattern.format(rep=rep)
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    lines = [
        "\\begin{tabular}{clllll}",
        "\\toprule",
        "$K$ & Dataset & prior & " + " & ".join(label for _, label in COLUMNS) + " \\\\",
        "\\midrule",
    ]
    for block, chains in enumerate([7, 5]):
        if block:
            lines.append("\\midrule")
        for ds in ["wdbc", "spambase", "adult"]:
            for prior in [0.1, 0.3]:
                key = {"dataset": ds, "prior": prior, "chains": chains}
                cells = [cell(ratios(rows, key, method)) for method, _ in COLUMNS]
                lines.append(f"{chains} & {DATASET_LABELS[ds]} & {prior} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
