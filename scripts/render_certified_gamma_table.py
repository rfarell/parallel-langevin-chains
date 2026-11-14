"""Render paper/tables/certified_gamma.tex from the S1 screen artifact.

Reads outputs/screens/certified_gamma_screen.jsonl (12 rows: six campaign
cells x {MAP, gold mean}) and emits the appendix table for rem:vr-mse.  No
recomputation: the renderer only formats the artifact's certified values,
asserts internal consistency (ratio == gamma_lb / threshold), and asserts
the 12 ratios against pinned three-significant-figure expectations from the
screen record (S1; promotion per protocol Amendment v2, sections A/B).  Any mismatch aborts and nothing is written.
"""

import json
import math
from pathlib import Path

SCREEN = Path("outputs/screens/certified_gamma_screen.jsonl")
OUT = Path("paper/tables/certified_gamma.tex")

CELLS = [("wdbc", 0.1), ("wdbc", 0.3), ("spambase", 0.1),
         ("spambase", 0.3), ("adult", 0.1), ("adult", 0.3)]
POINTS = ["map", "gold_mean"]
POINT_LABEL = {"map": "MAP", "gold_mean": "gold mean"}
DATASET_LABEL = {"wdbc": "WDBC", "spambase": "Spambase", "adult": "Adult"}
THRESHOLD = 0.03125  # 1/32

# Pinned expected ratios (3 significant figures) per (dataset, prior, point).
EXPECTED_RATIO = {
    ("wdbc", 0.1, "map"): 2.31e7, ("wdbc", 0.1, "gold_mean"): 1.79e8,
    ("wdbc", 0.3, "map"): 2.90e7, ("wdbc", 0.3, "gold_mean"): 4.05e7,
    ("spambase", 0.1, "map"): 5.07e6, ("spambase", 0.1, "gold_mean"): 7.58e8,
    ("spambase", 0.3, "map"): 1.34e6, ("spambase", 0.3, "gold_mean"): 1.73e8,
    ("adult", 0.1, "map"): 1.61e9, ("adult", 0.1, "gold_mean"): 4.99e12,
    ("adult", 0.3, "map"): 1.11e9, ("adult", 0.3, "gold_mean"): 2.06e15,
}


def fail(msg):
    raise SystemExit(f"render_certified_gamma_table: {msg}")


def sci(x):
    """$a.bc\\times10^{e}$ (3 significant figures)."""
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    if round(m, 2) >= 10.0:
        m /= 10.0
        e += 1
    return f"${m:.2f}\\times10^{{{e}}}$"


def main():
    rows = {}
    for line in SCREEN.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "screen" or r.get("screen") != "S1_certified_gamma":
            fail(f"unexpected record kind {r.get('kind')!r}/{r.get('screen')!r}")
        key = (r["dataset"], r["prior"], r["point"])
        if key in rows:
            fail(f"duplicate row {key}")
        rows[key] = r
    if len(rows) != 12:
        fail(f"expected 12 rows, found {len(rows)}")

    for key, r in rows.items():
        if r["threshold"] != THRESHOLD:
            fail(f"{key}: threshold {r['threshold']!r} != {THRESHOLD}")
        recomputed = r["gamma_lb"] / THRESHOLD
        if abs(recomputed - r["ratio_to_threshold"]) > 1e-9 * r["ratio_to_threshold"]:
            fail(f"{key}: ratio {r['ratio_to_threshold']!r} != gamma_lb/threshold")
        exp = EXPECTED_RATIO[key]
        if abs(r["ratio_to_threshold"] - exp) > 0.005 * exp:
            fail(f"{key}: ratio {r['ratio_to_threshold']!r} != pinned {exp}")

    ratios = [r["ratio_to_threshold"] for r in rows.values()]
    lo_key = min(rows, key=lambda k: rows[k]["ratio_to_threshold"])
    hi_key = max(rows, key=lambda k: rows[k]["ratio_to_threshold"])
    if lo_key != ("spambase", 0.3, "map") or hi_key != ("adult", 0.3, "gold_mean"):
        fail(f"extreme cells moved: min {lo_key}, max {hi_key}")
    if not (min(ratios) >= 1.3e6 and 2.0e15 < max(ratios) < 2.1e15):
        fail(f"remark-quoted extremes violated: min {min(ratios)!r}, max {max(ratios)!r}")

    lines = [
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Dataset & prior & point & $d$ & $m_{\\mathrm{ub}}$ & $L_{\\mathrm{lb}}$ & "
        "$\\rho_{\\mathrm{lb}}$ & $\\Gamma_{\\mathrm{lb}}$ & "
        "$\\Gamma_{\\mathrm{lb}}/(1/32)$ \\\\",
        "\\midrule",
    ]
    for ds, prior in CELLS:
        for point in POINTS:
            r = rows[(ds, prior, point)]
            lines.append(
                f"{DATASET_LABEL[ds]} & {prior} & {POINT_LABEL[point]} & {r['dim']} & "
                f"{r['m_ub']:.3g} & {r['L_lb']:.3g} & {r['rho_lb']:.3g} & "
                f"{sci(r['gamma_lb'])} & {sci(r['ratio_to_threshold'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} (12 rows; ratios {min(ratios):.3e}-{max(ratios):.3e})")


if __name__ == "__main__":
    main()
