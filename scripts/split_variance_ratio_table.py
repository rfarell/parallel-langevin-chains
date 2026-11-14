"""Split paper/tables/variance_ratio_check.tex into two page-sized parts.

The canonical single-file table (192 data rows, written by
scripts/variance_ratio_check.py --make-table and kept byte-identical) is too
tall for one float ("Float too large for page by 1381pt", a pre-existing
rendering defect that clipped the product family entirely).  This splitter
derives two paired-column presentation files from it, one per family:

  paper/tables/variance_ratio_check_product.tex  (96 rows as 48+48 side by side)
  paper/tables/variance_ratio_check_dense.tex    (96 rows as 48+48 side by side)

No numbers are generated here.  Self-checks (abort, nothing written):
exactly 96 rows per family; the emitted data rows, re-concatenated in family
order, are byte-identical to the source rows; header parsed verbatim.
"""

from pathlib import Path

SRC = Path("paper/tables/variance_ratio_check.tex")
OUT = {
    "product": Path("paper/tables/variance_ratio_check_product.tex"),
    "dense": Path("paper/tables/variance_ratio_check_dense.tex"),
}
COLSPEC_LINE = "\\begin{tabular}{lrrrrrrr}"


def fail(msg):
    raise SystemExit(f"split_variance_ratio_table: {msg}")


def main():
    lines = SRC.read_text().splitlines()
    if lines[0] != COLSPEC_LINE or lines[1] != "\\toprule" or lines[3] != "\\midrule":
        fail(f"unexpected preamble in {SRC}")
    header = lines[2]
    if not header.startswith("family &"):
        fail(f"unexpected header {header!r}")
    end = lines.index("\\bottomrule")
    rows = lines[4:end]
    by_family = {"product": [], "dense": []}
    for row in rows:
        fam = row.split(" &", 1)[0]
        if fam not in by_family:
            fail(f"unknown family in row {row!r}")
        by_family[fam].append(row)
    for fam, r in by_family.items():
        if len(r) != 96:
            fail(f"{fam}: {len(r)} rows != 96")
    # Byte-identity: source rows are dense-then-product contiguous blocks.
    if by_family["dense"] + by_family["product"] != rows:
        fail("re-concatenated rows differ from the source (order or content)")

    def block(rs):
        return "\n".join([COLSPEC_LINE, "\\toprule", header, "\\midrule",
                          *rs, "\\bottomrule", "\\end{tabular}"])

    for fam, out_path in OUT.items():
        r = by_family[fam]
        text = block(r[:48]) + "\\hspace{2em}%\n" + block(r[48:]) + "\n"
        out_path.write_text(text)
        print(f"wrote {out_path} ({len(r)} rows as 48+48)")


if __name__ == "__main__":
    main()
