"""Emit the three gold-endtask paper tables (P4 integration, 2026-07-18).

Layout: hardened dispersed-start reference PRIMARY with the per-cell
simultaneous 95% gold-error radius r95; the archived common-start
reference plus fresh-seed rescoring and the original table byte-preserved
in the appendix; the full 48-pair both-reference ordering grid.  Protocol:
campaigns/screens_precommit.md (V3-5 paired rescoring + the 2026-07-18
correction amendment).

Outputs:
  paper/tables/gold_endtask.tex           hardened primary (r95 column)
  paper/tables/gold_endtask_archived.tex  original block (byte-preserved)
                                          + fresh-at-archived block
  paper/tables/gold_ordering_grid.tex     48-pair archived->hardened labels

MANDATORY self-checks (any mismatch aborts, nothing is written):
  * recomputed 48-pair grid == outputs/screens/ordering_evaluation_v2_2026-07-18.txt
  * printed ranges == the pinned expectations (iid 0.845-10.185,
    C* 0.0135-0.1070, fine 0.0155-1.6350, MALA 0.0933-0.5310,
    VI/EP 0.00304-0.04184, MAP 6.67-245.28); r95 == the six pinned values
  * the regenerated original block byte-matches the previously committed
    paper/tables/gold_endtask.tex (sha256 pinned below)
  * every caption/body number quoted in paper/main.tex around
    tab:gold-endtask* is recomputed and asserted here.
Floor conventions: archived 5*s_c on
squared errors (s_c = archived gold_spread, itself squared); hardened
25*r95^2 (five-r95 distance rule).
"""

import hashlib
import json
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gold_rescore_p4 as gr  # noqa: E402  (strict loader + gold selection)
import p4_gate_analysis as pga  # noqa: E402  (q_P4 + r95 formula)
import summarize_mala_law_and_gold as smg  # noqa: E402  (archived formatting)

CELLS = [("wdbc", 0.1), ("wdbc", 0.3), ("spambase", 0.1),
         ("spambase", 0.3), ("adult", 0.1), ("adult", 0.3)]
ARMS = ["iid_eta1", "zs_eta1", "zs_eta2", "zs_mala"]
DATASET_LABEL = smg.DATASET_LABEL

RESCORE = Path("outputs/sweeps/gold_rescore_p4.jsonl")
ARCHIVED_GOLD = Path("outputs/sweeps/rr_extrapolation_gold.jsonl")
HARDENED_GOLD = Path("outputs/sweeps/rr_extrapolation_gold_p4_fd354ef.jsonl")
RR_REPS = [Path(f"outputs/sweeps/rr_extrapolation_rep{r}.jsonl") for r in range(3)]
VI_BASELINE = Path("outputs/sweeps/vi_baseline.jsonl")
MAP_BASELINE = Path("outputs/sweeps/map_baseline.jsonl")
GRID_RECORD = Path("outputs/screens/ordering_evaluation_v2_2026-07-18.txt")

OUT_PRIMARY = Path("paper/tables/gold_endtask.tex")
OUT_ARCHIVED = Path("paper/tables/gold_endtask_archived.tex")
OUT_GRID = Path("paper/tables/gold_ordering_grid.tex")

# sha256 of paper/tables/gold_endtask.tex as committed before this overhaul
# (the archived original block must reproduce it byte-for-byte).
ORIGINAL_TABLE_SHA256 = "faf7c4e627a6be48759c41205c7b40fd9227760e84130a5525691587acc9916c"

# Pinned expectations (rescore result record, 2026-07-18).
R95_EXPECTED = [0.125352, 0.094848, 0.127625, 0.102190, 0.246969, 0.152369]
RANGE_EXPECTED = {  # (min, max, decimals) over the six hardened cell medians
    "iid_eta1": (0.845, 10.185, 3),
    "zs_eta1": (0.0135, 0.1070, 4),
    "zs_eta2": (0.0155, 1.6350, 4),
    "zs_mala": (0.0933, 0.5310, 4),
}
VIEP_EXPECTED = (0.00304, 0.04184, 5)
MAP_EXPECTED = (6.67, 245.28, 2)


def fail(msg):
    raise SystemExit(f"make_gold_endtask_tables: SELF-CHECK FAILED: {msg}")


def check(cond, msg):
    if not cond:
        fail(msg)


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def sq_err(mu, gold_mean):
    mu = np.asarray(mu, dtype=np.float64)
    g = np.asarray(gold_mean, dtype=np.float64)
    return float(np.sum((mu - g) ** 2))


def ordering_label(e_sampler, e_baseline, floor, baseline_name, arm_name):
    """Registered rule: ordered only if >=1.5x AND both errors exceed the
    floor; floor failure -> U; floors pass but <1.5x -> N."""
    if not (e_sampler > floor and e_baseline > floor):
        return "U"
    lo, hi = sorted((e_sampler, e_baseline))
    if hi / lo < 1.5:
        return "N"
    return baseline_name if e_baseline < e_sampler else arm_name


def main():
    # --- load references + strict rescore -----------------------------------
    meta = next(r for r in load_jsonl(RESCORE) if r.get("kind") == "rescore_run_meta")
    arch = gr.load_archived_gold(ARCHIVED_GOLD)
    hard = gr.select_hardened_gold(HARDENED_GOLD, set(CELLS), meta["passing_attempts"])
    expected_ids = {c: {"archived": arch[c]["gold_id"], "hardened": hard[c]["gold_id"]}
                    for c in CELLS}
    fits, scores = gr.load_rescore(RESCORE, expected_gold_ids=expected_ids)
    check(len(fits) == 72, f"expected 72 fits, got {len(fits)}")

    q = pga.q_p4()
    r95 = {c: pga.r95_from_mcse(hard[c]["mcse"], q) for c in CELLS}
    for c, exp in zip(CELLS, R95_EXPECTED):
        check(round(r95[c], 6) == exp, f"r95[{c}] = {r95[c]!r} != pinned {exp}")

    # --- per (cell, arm, reference) replicate values and medians ------------
    vals = {}
    for sfid, fit in fits.items():
        c = (fit["dataset"], fit["prior"])
        for ref, gid in (("archived", arch[c]["gold_id"]), ("hardened", hard[c]["gold_id"])):
            vals.setdefault((c, fit["arm_name"], ref), []).append(
                scores[sfid][gid]["mse_gold_median"])
    for key, v in vals.items():
        check(len(v) == 3, f"{key}: {len(v)} replicates != 3")
    med = {k: statistics.median(v) for k, v in vals.items()}

    for arm, (lo, hi, nd) in RANGE_EXPECTED.items():
        m = [med[(c, arm, "hardened")] for c in CELLS]
        check(round(min(m), nd) == lo and round(max(m), nd) == hi,
              f"{arm} hardened median range ({min(m)!r}, {max(m)!r}) != pinned ({lo}, {hi})")

    # --- deterministic fits rescored against both references ----------------
    vi_rows, ep_rows, map_rows = {}, {}, {}
    for r in load_jsonl(VI_BASELINE):
        c = (r["dataset"], r["prior"])
        if r.get("kind") == "vi_baseline":
            vi_rows[c] = r
        elif r.get("kind") == "ep_baseline" and r.get("converged"):
            ep_rows[c] = r
    for r in load_jsonl(MAP_BASELINE):
        map_rows[(r["dataset"], r["prior"])] = r
    check(all(c in vi_rows and c in ep_rows and c in map_rows for c in CELLS),
          "missing VI/EP/MAP rows")

    fit_err = {}  # (cell, name, ref) -> squared error
    for c in CELLS:
        for name, mu, arch_field, row in (
                ("VI", vi_rows[c]["mu_vi"], "vi_err_sq_gold", vi_rows[c]),
                ("EP", ep_rows[c]["mu_ep"], "ep_err_sq_gold", ep_rows[c]),
                ("MAP", map_rows[c]["theta_map"], "map_err_sq_gold", map_rows[c])):
            ea = sq_err(mu, arch[c]["gold_mean"])
            check(abs(ea - row[arch_field]) <= 1e-9 * max(1.0, row[arch_field]),
                  f"{name}@{c}: recomputed archived error {ea!r} != artifact {row[arch_field]!r}")
            fit_err[(c, name, "archived")] = ea
            fit_err[(c, name, "hardened")] = sq_err(mu, hard[c]["gold_mean"])

    viep_h = [fit_err[(c, n, "hardened")] for c in CELLS for n in ("VI", "EP")]
    lo, hi, nd = VIEP_EXPECTED
    check(round(min(viep_h), nd) == lo and round(max(viep_h), nd) == hi,
          f"VI/EP hardened range ({min(viep_h)!r}, {max(viep_h)!r}) != pinned ({lo}, {hi})")
    map_h = [fit_err[(c, "MAP", "hardened")] for c in CELLS]
    lo, hi, nd = MAP_EXPECTED
    check(round(min(map_h), nd) == lo and round(max(map_h), nd) == hi,
          f"MAP hardened range ({min(map_h)!r}, {max(map_h)!r}) != pinned ({lo}, {hi})")

    # --- caption/body numbers quoted in main.tex ----------------------------
    # VI/EP agree to 2% on every cell (both references).
    for c in CELLS:
        for ref in ("archived", "hardened"):
            v, e = fit_err[(c, "VI", ref)], fit_err[(c, "EP", ref)]
            check(abs(v - e) / min(v, e) <= 0.02, f"VI/EP disagree >2% at {c} ({ref})")
    # VI/EP raw squared errors numerically lower than fresh C* ULA on all six
    # cells; descriptive ratio range prints as 1.9-6.5.
    ratios = []
    for c in CELLS:
        for n in ("VI", "EP"):
            check(fit_err[(c, n, "hardened")] < med[(c, "zs_eta1", "hardened")],
                  f"{n} not lower than C* at {c} under hardened gold")
            ratios.append(med[(c, "zs_eta1", "hardened")] / fit_err[(c, n, "hardened")])
    check(round(min(ratios), 1) == 1.9 and round(max(ratios), 1) == 6.5,
          f"descriptive ratio range ({min(ratios)!r}, {max(ratios)!r}) != 1.9-6.5")
    # Caption's descriptive at-reference-precision framing (F3 ruling): every
    # C* ULA cell median below its 25*r95^2 floor, every iid median above it.
    for c in CELLS:
        floor_h = 25.0 * r95[c] ** 2
        check(med[(c, "zs_eta1", "hardened")] < floor_h,
              f"C* cell median not below the hardened floor at {c}")
        check(med[(c, "iid_eta1", "hardened")] > floor_h,
              f"iid cell median not above the hardened floor at {c}")
    # Historical spambase/0.3 separation: original 1.7x; fresh 1.22x (both fits).
    sp = ("spambase", 0.3)
    orig_rows = [r for p in RR_REPS for r in load_jsonl(p)]
    orig_sp = [r["mse_gold_median"] for r in orig_rows
               if r.get("kind") == "estimator" and r.get("pair") == "primary"
               and r.get("estimator") == "single" and r.get("coupling") == "zero_sum"
               and r.get("level") == "eta1" and (r["dataset"], r["prior"]) == sp]
    orig_sp_med = statistics.median(orig_sp)
    fresh_sp_med = med[(sp, "zs_eta1", "archived")]
    check(round(orig_sp_med, 5) == 0.00989, f"original spambase/0.3 C* {orig_sp_med!r}")
    check(round(fresh_sp_med, 4) == 0.0139, f"fresh spambase/0.3 C* {fresh_sp_med!r}")
    for n, fresh_exp, orig_exp in (("VI", 1.225, 1.719), ("EP", 1.221, 1.714)):
        check(round(fit_err[(sp, n, "archived")] / fresh_sp_med, 3) == fresh_exp,
              f"fresh archived ratio vs {n} != {fresh_exp}")
        check(round(fit_err[(sp, n, "archived")] / orig_sp_med, 3) == orig_exp,
              f"original archived ratio vs {n} != {orig_exp}")
    # Paired hardened/archived contrast (ratio of cell medians).
    paired = {arm: [med[(c, arm, "hardened")] / med[(c, arm, "archived")] for c in CELLS]
              for arm in ARMS}
    allp = [x for v in paired.values() for x in v]
    check(round(min(allp), 3) == 0.899 and round(max(allp), 3) == 1.149,
          f"paired sampler ratio range ({min(allp)!r}, {max(allp)!r}) != 0.899-1.149")
    check(round(med[(sp, "zs_eta1", "hardened")] / med[(sp, "zs_eta1", "archived")], 3) == 0.975,
          "spambase/0.3 paired C* ratio != 0.975")
    # VI/EP rescoring moves: lower on exactly five cells, higher on adult/0.3.
    for n in ("VI", "EP"):
        drops = [c for c in CELLS if fit_err[(c, n, "hardened")] < fit_err[(c, n, "archived")]]
        check(len(drops) == 5 and ("adult", 0.3) not in drops, f"{n} drop pattern != 5/6")
    check(round(fit_err[(sp, "VI", "archived")], 4) == 0.0170
          and round(fit_err[(sp, "VI", "hardened")], 4) == 0.0031,
          "spambase/0.3 VI rescore != 0.0170->0.0031")
    check(round(fit_err[(("adult", 0.3), "VI", "archived")], 4) == 0.0240
          and round(fit_err[(("adult", 0.3), "VI", "hardened")], 4) == 0.0418,
          "adult/0.3 VI rescore != 0.0240->0.0418")
    # Stationary-bias claims: original seeds below the archived spread in
    # every replicate; fresh seeds inside r95 (bias_sq < r95^2) under the
    # hardened reference in every replicate.  (The fresh-seed/archived-spread
    # combination is NOT claimed anywhere: one wdbc/0.1 replicate exceeds it.)
    for r in orig_rows:
        if (r.get("kind") == "estimator" and r.get("pair") == "primary"
                and r.get("estimator") == "single" and r.get("coupling") == "zero_sum"
                and r.get("level") == "eta1"):
            check(r["bias_sq_gold"] < r["gold_spread"],
                  f"original-seed bias claim fails at {(r['dataset'], r['prior'])}")
    for sfid, fit in fits.items():
        if fit["arm_name"] != "zs_eta1":
            continue
        c = (fit["dataset"], fit["prior"])
        check(scores[sfid][hard[c]["gold_id"]]["bias_sq_gold"] < r95[c] ** 2,
              f"fresh-seed hardened bias claim fails at {c} rep {fit['seed_idx']}")

    # --- 48-pair ordering grid, both references -----------------------------
    grid = {}
    for c in CELLS:
        floor_a = 5.0 * arch[c]["gold_spread"]      # archived rule (squared)
        floor_h = 25.0 * r95[c] ** 2                # hardened five-r95 rule
        for arm in ARMS:
            for n in ("VI", "EP"):
                la = ordering_label(med[(c, arm, "archived")],
                                    fit_err[(c, n, "archived")], floor_a, n, arm)
                lh = ordering_label(med[(c, arm, "hardened")],
                                    fit_err[(c, n, "hardened")], floor_h, n, arm)
                grid[(c, arm, n)] = (la, lh)
    check(all(lh == "U" for _, lh in grid.values()),
          "not all 48 pairs floor-unresolved under the hardened reference")
    # Byte-level agreement with the corrected evaluation record.
    recorded = {}
    for line in GRID_RECORD.read_text().splitlines():
        if ":" not in line or "-VI:" not in line:
            continue
        cell_tag, rest = line.split(":", 1)
        ds, pr = cell_tag.split("/")
        c = (ds, float(pr))
        for tok in rest.split():
            pair, labels = tok.rsplit(":", 1)
            arm, n = pair.rsplit("-", 1)
            la, lh = labels.split("->")
            recorded[(c, arm, n)] = (la, lh)
    check(len(recorded) == 48, f"evaluation record has {len(recorded)} pairs != 48")
    for key, lab in grid.items():
        check(recorded.get(key) == lab,
              f"grid mismatch at {key}: recomputed {lab} != recorded {recorded.get(key)}")

    # --- original table block: regenerate + byte-compare --------------------
    with tempfile.NamedTemporaryFile("r", suffix=".tex", delete=False) as tf:
        tmp_path = Path(tf.name)
    smg.gold_table(orig_rows + [arch[c] for c in CELLS], tmp_path,
                   map_rows=[map_rows[c] for c in CELLS],
                   vi_rows=[vi_rows[c] for c in CELLS] + [ep_rows[c] for c in CELLS])
    original_text = tmp_path.read_text()
    tmp_path.unlink()
    got_sha = hashlib.sha256(original_text.encode()).hexdigest()
    check(got_sha == ORIGINAL_TABLE_SHA256,
          f"regenerated original table sha256 {got_sha} != pinned {ORIGINAL_TABLE_SHA256}")
    original_lines = original_text.splitlines()
    header_idx = original_lines.index("\\midrule")
    orig_data_rows = original_lines[header_idx + 1:original_lines.index("\\bottomrule")]
    check(len(orig_data_rows) == 6, "original table does not have 6 data rows")

    # --- emit: hardened primary ---------------------------------------------
    fmt = "{:.3g}"
    lines = [
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Dataset & prior & MAP & VI & EP & iid ULA & $C_\\star$ ULA & "
        "$C_\\star$ ULA (fine) & MALA & $r_{95}$ \\\\",
        "\\midrule",
    ]
    for c in CELLS:
        ds, prior = c
        parts = [f"{DATASET_LABEL[ds]} & {prior}",
                 fmt.format(fit_err[(c, 'MAP', 'hardened')]),
                 fmt.format(fit_err[(c, 'VI', 'hardened')]),
                 fmt.format(fit_err[(c, 'EP', 'hardened')])]
        for arm in ARMS:
            parts.append(smg.med_range(vals[(c, arm, "hardened")], fmt))
        parts.append(fmt.format(r95[c]))
        lines.append(" & ".join(parts) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    OUT_PRIMARY.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PRIMARY}")

    # --- emit: archived (original block byte-preserved + fresh block) -------
    lines = original_lines[:header_idx + 1]
    lines.append("\\multicolumn{10}{l}{\\emph{Original sampler seeds "
                 "(as previously reported)}} \\\\")
    lines += orig_data_rows
    lines.append("\\midrule")
    lines.append("\\multicolumn{10}{l}{\\emph{Fresh rescore seeds, scored "
                 "against the same archived reference}} \\\\")
    for c in CELLS:
        ds, prior = c
        parts = [f"{DATASET_LABEL[ds]} & {prior}",
                 fmt.format(fit_err[(c, 'MAP', 'archived')]),
                 fmt.format(fit_err[(c, 'VI', 'archived')]),
                 fmt.format(fit_err[(c, 'EP', 'archived')])]
        for arm in ARMS:
            parts.append(smg.med_range(vals[(c, arm, "archived")], fmt))
        parts.append(f"{arch[c]['gold_spread']:.1e}")
        lines.append(" & ".join(parts) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    OUT_ARCHIVED.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_ARCHIVED}")

    # --- emit: 48-pair ordering grid ----------------------------------------
    arm_head = {"iid_eta1": "iid", "zs_eta1": "$C_\\star\\eta_1$",
                "zs_eta2": "$C_\\star\\eta_2$", "zs_mala": "MALA"}
    heads = [f"{arm_head[arm]}--{n}" for arm in ARMS for n in ("VI", "EP")]
    lines = [
        "\\begin{tabular}{ll" + "c" * 8 + "}",
        "\\toprule",
        "Dataset & prior & " + " & ".join(heads) + " \\\\",
        "\\midrule",
    ]
    for c in CELLS:
        ds, prior = c
        cells_out = [f"{la}$\\to${lh}" for arm in ARMS for n in ("VI", "EP")
                     for la, lh in [grid[(c, arm, n)]]]
        lines.append(f"{DATASET_LABEL[ds]} & {prior} & " + " & ".join(cells_out) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    OUT_GRID.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_GRID}")

    # --- caption-number block (for the main.tex captions; informational) ----
    print("\nCAPTION NUMBERS (all asserted above):")
    for arm in ARMS:
        m = [med[(c, arm, "hardened")] for c in CELLS]
        print(f"  hardened {arm} medians: min {min(m):.6g} max {max(m):.6g}")
    print(f"  VI/EP hardened: min {min(viep_h):.6g} max {max(viep_h):.6g}; "
          f"MAP hardened: min {min(map_h):.6g} max {max(map_h):.6g}")
    print(f"  descriptive C*/fit ratios: {min(ratios):.3f}-{max(ratios):.3f}")
    print(f"  spambase/0.3 C*: original {orig_sp_med:.6g} -> fresh {fresh_sp_med:.6g}; "
          f"fresh ratios VI {fit_err[(sp, 'VI', 'archived')] / fresh_sp_med:.3f} "
          f"EP {fit_err[(sp, 'EP', 'archived')] / fresh_sp_med:.3f}")
    print(f"  paired hardened/archived medians: {min(allp):.4f}-{max(allp):.4f} "
          f"(spambase/0.3 C*: {med[(sp, 'zs_eta1', 'hardened')] / med[(sp, 'zs_eta1', 'archived')]:.3f})")
    print(f"  r95: {[round(r95[c], 6) for c in CELLS]}")
    print("all self-checks passed")


if __name__ == "__main__":
    main()
