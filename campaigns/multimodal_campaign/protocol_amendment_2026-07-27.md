# Bimodal mode-occupancy study: protocol amendment of 2026-07-27

Verbatim text of the dated amendment to the pre-registered protocol of the
bimodal mixture study (Appendix G of the paper, mode-occupancy table).  The
amendment was written after the campaign ran and before any scored metric
was viewed.

## Amendment (2026-07-27, post-run, emitter-side, before any scored
## metric was viewed)

The first emitter run aborted on the FPT gate for the common arm in all
six stationary cells (fpt-median c0–c3, c5, c6 common); every other gate
passed (125/131) and the validator was clean. Diagnosis, run on FPT
statistics ONLY (no scored metric was computed or seen): the emitter had
operationalized "first-passage-time quartiles match iid" as the
across-ensemble mean of the WITHIN-ensemble median of the K chains' FPTs.
That statistic is nonlinear in the joint law. Under common noise the K
chains coalesce and cross near-simultaneously, so the within-ensemble
median approaches a single chain's draw, whose expectation exceeds the
population median for a right-skewed law; the statistic's expectation
therefore shifts under within-ensemble dependence even when every chain's
marginal law is identical — and the marginal law is the quantity
Proposition 1 fixes. Measured pooled per-chain quartiles (the
dependence-free object): cell 0 medians 45/46/47/47 and cell 2 medians
8188/8080/8120/8111 across iid/zero_sum/antithetic/common — identical
within Monte Carlo error — while the per-ensemble-median means shifted
only under common (58/59/61/75 in cell 0; 10504/10324/10901/13621 in
cell 2). The linear-statistic gates (transition counts, second moments,
occupancy means) passed for every arm, localizing the failure to the gate
statistic.

Correction: the FPT gate statistic becomes the per-ensemble MEAN of
uncensored per-chain FPTs — linear in the per-chain values, so its
expectation is invariant to within-ensemble dependence — under the same
4-sigma across-ensemble machinery ("fpt-mean" in the emitter). Pooled
per-chain quartiles are additionally reported descriptively per arm. No
dynamics row is touched; the emitter re-runs read-only over the same
JSONL. Rows remain bound to the protocol as locked at commit d4fd741, and
the emitter verifies each row's precommit_sha256 against that commit's
file bytes (this amendment changes the living file's hash, not the locked
one).
