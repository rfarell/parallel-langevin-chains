# How Should Parallel Langevin Chains Share Noise?

Code and raw results for the NeurIPS 2026 paper

> Ryan Farell. *How Should Parallel Langevin Chains Share Noise?*
> Advances in Neural Information Processing Systems 39 (NeurIPS 2026).
> [Paper and reviews on OpenReview](https://openreview.net/forum?id=dTLaD3Us8h)

Parallel Langevin samplers usually give every chain its own Gaussian noise.
The unadjusted Langevin algorithm (ULA) fixes only the distribution of each
chain's noise, so the draws may depend on one another across chains.
Subtracting the across-chain mean from every draw and rescaling leaves each
chain's path law unchanged and removes the noise from the ensemble mean. The
paper derives what this does to estimates computed from the ensemble mean and
measures it on synthetic targets and real posteriors.

This repository contains the scripts that ran the experiments, the raw rows
they produced, and the tables as printed. Table and figure numbers below are
those of the camera-ready paper.

## The coupling

The whole method is one line of arithmetic on the noise array
(`scripts/zero_sum_langevin.py`, excerpt):

```python
def make_noise(shape, method, dtype=cp.float32):
    reps, chains, dim = shape              # ensembles x chains x coordinates
    if method == "iid":
        return cp.random.standard_normal(shape, dtype=dtype)
    if method == "zero_sum":
        z = cp.random.standard_normal(shape, dtype=dtype)
        z = z - z.mean(axis=1, keepdims=True)           # subtract the mean over chains
        return z * math.sqrt(chains / (chains - 1))     # restore unit variance
    ...
```

The scripts use four couplings throughout:

| Name in code | Name in the paper | Construction |
| --- | --- | --- |
| `iid` | iid | independent draws |
| `zero_sum` | centering, $C_\star$ | subtract the mean over chains, rescale |
| `antithetic` | pairing | sign-flipped pairs; with an odd chain count one draw is unmatched |
| `common` | common noise | one draw shared by all chains |

## Layout

| Path | Contents |
| --- | --- |
| `scripts/` | Experiment runners, table and figure generators, CPU checks. |
| `outputs/sweeps/` | Raw JSONL rows, one file per run. Each row is one configuration with its settings, seeds and results. |
| `campaigns/` | Six campaigns kept with their own launchers and runners: `attempt01` (8 versus 128 chains), `attempt02` (odd chain counts), `attempt03` and `minibatch_followup*` (minibatch SGLD), `attempt08` (variance-time frontier), `zv_campaign` (control variates), `multimodal_campaign` (bimodal target). |
| `outputs/reports/`, `outputs/screens/` | Logs and records the paper cites: the NUTS self-test log, the 1D check of the rejection law, the certified-$\Gamma$ screen and the reference-ordering record. |
| `paper/tables/` | The LaTeX tables as printed. File names match the appendix. |

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt     # CuPy (CUDA 12), NumPy, SciPy, Matplotlib
.venv/bin/python -m pip install torch==2.5.1            # only for the ZV and bimodal studies and their table renderer
```

The sweeps ran with Python 3.9 on NVIDIA A100 40 GB GPUs (CuPy 13.6, CUDA
12.2). The ZV and bimodal studies ran with PyTorch 2.5.1 on an NVIDIA GB10
system. Table and figure generation runs on a CPU. Run every command from the
repository root.

## Quick start on a CPU

```bash
.venv/bin/python scripts/theory_sanity_checks.py       # five identities and bounds from the theory; prints PASS lines
.venv/bin/python scripts/plot_intro_effect.py          # Figure 1, right panel
.venv/bin/python scripts/experiment_diagnostics.py     # Tables 12, 13, 15 and three supplementary diagnostic tables
git diff --stat paper/tables                           # what changed against the printed tables
```

These three need only NumPy and Matplotlib. On a machine without CUDA, install
`numpy==2.0.2 scipy==1.13.1 matplotlib==3.9.4` instead of `requirements.txt`.

## Reproducing the paper

There are two levels. Level 1 rebuilds tables and figures from the rows in
this repository and takes minutes on a laptop. Level 2 reruns the experiments
on a GPU.

### Map from the paper to the code

| Paper | Study | Rows | Table or figure generator | Experiment runner |
| --- | --- | --- | --- | --- |
| Fig. 1 | Ensemble mean follows gradient descent | computed on the fly | `plot_intro_effect.py` | same |
| Figs. 2, 3; Tables 7, 14 | Synthetic Gaussian, quartic and logistic targets; start factors | `shard{0,1,2}.jsonl`, `init_*_paired.jsonl` | `summarize_results.py`, `plot_results.py` | `zero_sum_langevin.py` |
| Fig. 6 | Crossover for a mixed Gaussian observable | `hermite_*_hi.jsonl` | `plot_hermite.py` | `hermite_observable.py` |
| Table 8 | Synthetic logistic posterior-mean error | `logistic_mse_dim*.jsonl` | `plot_logistic_mse.py` | `logistic_mse.py` |
| Tables 9, 36; Figs. 2, 4, 5 | Chain count and sampling cost on UCI posteriors | `campaign_synthetic_rep*.jsonl`, `campaign_real_full_rep*.jsonl` | `plot_campaign.py` | `wallclock_sweep.py`, `real_logistic_gpu.py` |
| Table 2 | Main-text summary | none | rounded from Tables 9 and 10 | none |
| Table 10 | Eight centered chains versus 128 iid chains | `campaigns/attempt01/` | `regen_highk_table.py`, `render_study_tables.py` | `campaigns/run_attempt01_*.sh` |
| Table 11 | Variance times sampling time | `campaigns/attempt08/` | `regen_frontier_table.py` | `campaigns/run_attempt08_*.sh` |
| Tables 12, 13, 15 | Checks of the covariance identity | computed on the fly | `experiment_diagnostics.py` | same |
| Table 16 | Odd chain counts on UCI posteriors | `campaigns/attempt02/` | `regen_oddk_table.py` | `campaigns/run_attempt02_rep.sh` |
| Table 17 | Pairs plus a centered triple | `oddk_zerosum_rep*.jsonl` | `oddk_zerosum_table.py` | `oddk_zerosum_baseline.py` |
| Table 18 | Minibatch SGLD | `campaigns/attempt03/`, `campaigns/minibatch_followup*/` | `regen_sgld_table.py` | `campaigns/run_attempt03_rep.sh`, `campaigns/run_minibatch_followup_*.sh` |
| Table 19 | MALA on UCI posteriors | `mala_real_rep*.jsonl` | `summarize_new_experiments.py` | `mala_real_logistic.py` |
| Table 20 | MALA step-size sensitivity | `mala_sensitivity.jsonl` | printed table only | `mala_real_logistic.py` |
| Table 6 | Rejection-leak law | `mala_diag_rep*.jsonl` | `summarize_mala_law_and_gold.py` | `mala_real_logistic.py`, `verify_mala_law_1d.py` |
| Tables 21, 22, 23 | Accuracy against a long MALA reference; VI, EP and MAP baselines | `rr_extrapolation_*.jsonl`, `gold_rescore_p4.jsonl`, `vi_baseline.jsonl`, `map_baseline.jsonl` | `make_gold_endtask_tables.py` | `rr_extrapolation.py`, `gold_rescore_p4.py`, `vi_baseline.py`, `map_baseline.py` |
| Table 24 | Fixed preconditioner | `preconditioned_real_rep*.jsonl` | `preconditioned_real.py --make-table` | `preconditioned_real.py` |
| Table 25 | HMC with shared momentum refresh | `hmc_real_rep*.jsonl` | `hmc_real_logistic.py --make-table` | `hmc_real_logistic.py` |
| Table 26 | Jittered path lengths and NUTS | `nuts_real_rep*.jsonl` | `nuts_real_logistic.py --make-table` | `nuts_real_logistic.py` |
| Table 27 | ChEES tuning on coupled chains | `chees_closedloop_rep*.jsonl` | printed table only | `chees_real_logistic.py` |
| Table 28 | Gaussian process hyperparameters | `gp_hyper_gold.jsonl`, `gp_hyper_rep*.jsonl` | `gp_hyper_real.py --make-table` | `gp_hyper_real.py` |
| Table 29 | Posterior uncertainty observables | `uncertainty_obs_rep*.jsonl` | `summarize_new_experiments.py` | `uncertainty_observables.py` |
| Table 30 | Student-t prior and hierarchical posteriors | `nonlogconcave_rep*.jsonl` | `summarize_new_experiments.py` | `nonlogconcave_real.py` |
| Tables 31, 32 | Zero-variance control variates | `campaigns/zv_campaign/` | `render_study_tables.py` | `zv_composition.py` |
| Table 33 | Mode occupancy on a bimodal mixture | `campaigns/multimodal_campaign/` | `render_study_tables.py` | `multimodal_bimodal.py` |
| Table 34 | Gisette and MNIST 7/9 | `highdim_real_rep*.jsonl` | `summarize_new_experiments.py` | `highdim_real.py` |
| Table 35 | Exact Bayesian linear regression | `bayes_linreg_rep*.jsonl` | `summarize_new_experiments.py` | `bayes_linreg_exact.py` |
| Table 3 | Certified lower bounds on $\Gamma$ | `outputs/screens/certified_gamma_screen.jsonl` | `render_certified_gamma_table.py` | `certified_gamma_table.py` |
| Tables 4, 5 | Check of the variance-ratio theorem | `variance_ratio_rep*.jsonl` | `variance_ratio_check.py --make-table`, `split_variance_ratio_table.py` | `variance_ratio_check.py` |
| App. G.10 | Step times across three GPUs | `distributed_overhead.jsonl` | none | `distributed_overhead.py` |

Rows without a directory live in `outputs/sweeps/`; scripts live in `scripts/`.
Four more tables accompany the paper without being printed in it:
`diagnostic_weighted_timeavg.tex`, `diagnostic_nonlinear.tex`,
`diagnostic_sgld.tex` (all from `experiment_diagnostics.py`) and
`odd_k_summary.tex` (from `plot_oddk.py`).

### Level 1: tables and figures from the included rows

Generators write into `paper/tables/` and `paper/figures/`. Afterwards
`git diff paper/tables` shows any difference from the printed tables.

```bash
PY=.venv/bin/python; S=outputs/sweeps

# Figures 2, 3 and Tables 7, 14
$PY scripts/summarize_results.py $S/shard{0,1,2}.jsonl $S/init_{short,mid,long}_paired.jsonl \
    --out paper/tables/summary.tex --init-out paper/tables/init_summary.tex
$PY scripts/plot_results.py $S/shard{0,1,2}.jsonl $S/init_{short,mid,long}_paired.jsonl --outdir paper/figures

# Figure 6
$PY scripts/plot_hermite.py $S/hermite_{odd,even,largek}_hi.jsonl \
    --fig paper/figures/hermite_observable_ratio.png --table paper/figures/hermite_summary.tex

# Table 8
$PY scripts/plot_logistic_mse.py $S/logistic_mse_dim{64,128,256}{,_rep{2..20}}.jsonl \
    --fig paper/figures/logistic_mse.png --table paper/tables/logistic_mse.tex

# Tables 9, 36 and Figures 2 (right panel), 4, 5
$PY scripts/plot_campaign.py $S/campaign_synthetic_rep{0,1,2}.jsonl $S/campaign_real_full_rep{0,1,2}.jsonl \
    --fig paper/figures/campaign_wallclock_metric_equivalent_iid.png \
    --real-k-fig paper/figures/campaign_real_k_sweep.png \
    --real-table paper/tables/campaign_real_summary.tex \
    --wallclock-table paper/tables/campaign_wallclock_summary.tex

# Tables 19, 29, 30, 34, 35
$PY scripts/summarize_new_experiments.py

# Table 6, then Tables 21-23. Keep this order: the first command also writes
# an older layout of Table 21, which the second command replaces.
$PY scripts/summarize_mala_law_and_gold.py
$PY scripts/make_gold_endtask_tables.py

# Tables 17 and 3
$PY scripts/oddk_zerosum_table.py
$PY scripts/render_certified_gamma_table.py

# Tables 24, 25, 26, 28, 4, 5 (these scripts import CuPy at start-up)
$PY scripts/preconditioned_real.py --make-table
$PY scripts/hmc_real_logistic.py --make-table
$PY scripts/nuts_real_logistic.py --make-table
$PY scripts/gp_hyper_real.py --make-table
$PY scripts/variance_ratio_check.py --make-table --ratio-constant 15.5895   # 512/(81 ln(3/2))
$PY scripts/split_variance_ratio_table.py

# Tables 10, 31, 32, 33 (needs PyTorch)
$PY scripts/render_study_tables.py

# Tables 10, 11, 16, 18: recompute from the campaign rows and require byte
# identity with the printed table. Nothing under paper/ is written.
$PY scripts/regen_highk_table.py --mode archived \
    --primary campaigns/attempt01/rep{3,4,5}.jsonl --stress campaigns/attempt01/stress_rep{6,7,8}.jsonl
$PY scripts/regen_oddk_table.py --mode archived
$PY scripts/regen_sgld_table.py --mode archived
$PY scripts/regen_frontier_table.py --mode archived --out-dir /tmp/regen_frontier
```

Regenerated tables match the printed ones. A few headers and row labels were
edited by hand for print, for example `zero-sum` became `$C_\star$`, and two
entries that are zero in exact arithmetic print round-off noise. Figures show
the same curves with small differences in font rendering. Several scripts
also have a `--selftest` mode.

### Level 2: rerunning the experiments

Download the datasets first. They are not redistributed here.

```bash
$PY scripts/download_real_data.py --dataset all     # WDBC, Spambase, Adult -> data/real/
$PY scripts/download_highdim_data.py                # Gisette, MNIST digits 7 and 9
$PY scripts/download_regression_data.py             # Superconductivity
```

Runners append JSONL rows to `--out`. Write reruns to a fresh directory, or
remove the target file first, so new rows do not mix with the included ones.
Most runners accept `--quick` for a short smoke run. The commands below are
the ones that produced the included rows; on a single GPU replace the device
index with `0`.

<details>
<summary>Synthetic targets (Figures 2, 3, 6; Tables 7, 8, 14)</summary>

```bash
for i in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$i $PY scripts/zero_sum_langevin.py --shard $i --out outputs/sweeps/shard$i.jsonl
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/zero_sum_langevin.py --shard 3 --out outputs/sweeps/init_short_paired.jsonl
CUDA_VISIBLE_DEVICES=1 $PY scripts/zero_sum_langevin.py --shard 4 --out outputs/sweeps/init_mid_paired.jsonl
CUDA_VISIBLE_DEVICES=2 $PY scripts/zero_sum_langevin.py --shard 5 --out outputs/sweeps/init_long_paired.jsonl

CUDA_VISIBLE_DEVICES=0 $PY scripts/hermite_observable.py --shard 0 --seed-offset 1000000 --out outputs/sweeps/hermite_odd_hi.jsonl
CUDA_VISIBLE_DEVICES=1 $PY scripts/hermite_observable.py --shard 1 --seed-offset 1000000 --out outputs/sweeps/hermite_even_hi.jsonl
CUDA_VISIBLE_DEVICES=2 $PY scripts/hermite_observable.py --shard 2 --seed-offset 1000000 --out outputs/sweeps/hermite_largek_hi.jsonl

# 20 replicates of the synthetic logistic validation
for rep in $(seq 1 20); do
  if [ "$rep" -eq 1 ]; then suffix=""; offset=0; else suffix="_rep${rep}"; offset=$((1000000 * (rep - 1))); fi
  CUDA_VISIBLE_DEVICES=0 $PY scripts/logistic_mse.py --shard 0 --seed-offset "$offset" --out "outputs/sweeps/logistic_mse_dim64${suffix}.jsonl"
  CUDA_VISIBLE_DEVICES=1 $PY scripts/logistic_mse.py --shard 1 --seed-offset "$offset" --out "outputs/sweeps/logistic_mse_dim128${suffix}.jsonl"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/logistic_mse.py --shard 2 --seed-offset "$offset" --out "outputs/sweeps/logistic_mse_dim256${suffix}.jsonl"
done

# Synthetic odd chain counts (supplementary table odd_k_summary.tex); its rows are not included
CUDA_VISIBLE_DEVICES=0 $PY scripts/zero_sum_langevin.py --shard 6 --out outputs/sweeps/odd_k_ablation.jsonl
$PY scripts/plot_oddk.py outputs/sweeps/odd_k_ablation.jsonl \
    --fig paper/figures/odd_k_ablation.png --table paper/tables/odd_k_summary.tex
```
</details>

<details>
<summary>Chain count and cost on UCI posteriors (Tables 9, 36; Figures 2, 4, 5)</summary>

```bash
for r in 0 1 2; do
  for target in gaussian quartic synthetic_logistic; do
    CUDA_VISIBLE_DEVICES=$r $PY scripts/wallclock_sweep.py --target $target --replicate $r \
        --out outputs/sweeps/campaign_synthetic_rep$r.jsonl
  done
  for dataset in wdbc spambase adult; do
    CUDA_VISIBLE_DEVICES=$r $PY scripts/real_logistic_gpu.py --dataset $dataset --replicate $r --chains 4 8 16 32 \
        --out outputs/sweeps/campaign_real_full_rep$r.jsonl
  done
done
```

The defaults use priors `0.1` and `0.3`. WDBC and Spambase run 900 steps with
450 discarded; Adult runs 700 with 350 discarded.
</details>

<details>
<summary>The four campaigns under <code>campaigns/</code> (Tables 10, 11, 16, 18)</summary>

Each launcher takes a replicate number, a GPU index and an output file, and
loops over the three datasets.

```bash
# Table 10: main run (replicates 3-5) and repeat (6-8)
for r in 3 4 5; do bash campaigns/run_attempt01_rep.sh        $r 0 rerun/attempt01/rep$r.jsonl; done
for r in 6 7 8; do bash campaigns/run_attempt01_stress_rep.sh $r 0 rerun/attempt01/stress_rep$r.jsonl; done

# Table 16: K=7 main run (9-14), K=7 repeat (15-20), K=5 (21-23)
for r in $(seq 9 14);  do bash campaigns/run_attempt02_rep.sh $r 0 7 rerun/attempt02/primary_rep$r.jsonl; done
for r in $(seq 15 20); do bash campaigns/run_attempt02_rep.sh $r 0 7 rerun/attempt02/stress_k7_rep$r.jsonl; done
for r in 21 22 23;     do bash campaigns/run_attempt02_rep.sh $r 0 5 rerun/attempt02/edge_k5_rep$r.jsonl; done

# Table 18: main run (30-35), repeat (36-41), half batch size (42-44),
# random starts with independent batches (60-65), one shared batch stream (70-72)
for r in $(seq 30 35); do bash campaigns/run_attempt03_rep.sh $r 0 primary rerun/attempt03/primary_rep$r.jsonl; done
for r in $(seq 36 41); do bash campaigns/run_attempt03_rep.sh $r 0 primary rerun/attempt03/stress_rep$r.jsonl; done
for r in 42 43 44;     do bash campaigns/run_attempt03_rep.sh $r 0 small   rerun/attempt03/edge_rep$r.jsonl; done
for r in $(seq 60 65); do bash campaigns/run_minibatch_followup_allcells_rep.sh      $r 0 rerun/allcells_rep$r.jsonl; done
for r in 70 71 72;     do bash campaigns/run_minibatch_followup_common_stream_rep.sh $r 0 rerun/common_stream_rep$r.jsonl; done

# Table 11: arguments are GPU, start scale, mode, block, output prefix, replicates
bash campaigns/run_attempt08_reps.sh 0 0.0  full primary rerun/attempt08/primary_rep $(seq 300 305)
bash campaigns/run_attempt08_reps.sh 0 0.0  full stress  rerun/attempt08/stress_rep  $(seq 306 311)
bash campaigns/run_attempt08_reps.sh 0 0.25 full rand    rerun/attempt08/rand_rep    $(seq 312 317)
bash campaigns/run_attempt08_reps.sh 0 1.0  full scale1  rerun/attempt08/scale1_rep  $(seq 318 323)
bash campaigns/run_attempt08_reps.sh 0 0.0  k5   k5      rerun/attempt08/k5_rep      $(seq 324 329)
```

</details>

<details>
<summary>Other transitions and targets (Tables 6, 17, 19, 24 to 26, 28 to 30, 34, 35; App. G.10)</summary>

```bash
for r in 0 1 2; do
  $PY scripts/mala_real_logistic.py --dataset all --kernel mala --calibrate --replicate $r --out outputs/sweeps/mala_real_rep$r.jsonl
  $PY scripts/mala_real_logistic.py --dataset all --kernel ula  --calibrate --replicate $r --out outputs/sweeps/mala_real_rep$r.jsonl
  $PY scripts/mala_real_logistic.py --dataset all --kernel mala --calibrate --replicate $r --out outputs/sweeps/mala_diag_rep$r.jsonl
  $PY scripts/preconditioned_real.py --replicate $r --out outputs/sweeps/preconditioned_real_rep$r.jsonl
  $PY scripts/hmc_real_logistic.py --dataset all --replicate $r --out outputs/sweeps/hmc_real_rep$r.jsonl
  $PY scripts/nuts_real_logistic.py --dataset all --replicate $r --out outputs/sweeps/nuts_real_rep$r.jsonl
  $PY scripts/oddk_zerosum_baseline.py --replicate $r --out outputs/sweeps/oddk_zerosum_rep$r.jsonl
  $PY scripts/bayes_linreg_exact.py --replicate $r --out outputs/sweeps/bayes_linreg_rep$r.jsonl
  $PY scripts/uncertainty_observables.py --replicate $r --out outputs/sweeps/uncertainty_obs_rep$r.jsonl
  $PY scripts/nonlogconcave_real.py --model tprior --replicate $r --out outputs/sweeps/nonlogconcave_rep$r.jsonl
  $PY scripts/nonlogconcave_real.py --model hierarchical --calibrate --replicate $r --out outputs/sweeps/nonlogconcave_rep$r.jsonl
  $PY scripts/highdim_real.py --dataset mnist79 --replicate $r --out outputs/sweeps/highdim_real_rep$r.jsonl
  $PY scripts/highdim_real.py --dataset gisette --replicate $r --out outputs/sweeps/highdim_real_rep$r.jsonl
  $PY scripts/gp_hyper_real.py --mode ula --replicate $r --out outputs/sweeps/gp_hyper_rep$r.jsonl
done
$PY scripts/gp_hyper_real.py --selftest
$PY scripts/gp_hyper_real.py --mode gold --out outputs/sweeps/gp_hyper_gold.jsonl
$PY scripts/nuts_real_logistic.py --selftest      # archived log: outputs/reports/nuts_selftest_2026-07-15.txt
$PY scripts/distributed_overhead.py --out outputs/sweeps/distributed_overhead.jsonl   # three GPUs on one node, NCCL
```
</details>

<details>
<summary>Reference accuracy, the theorem check and the certified bound (Tables 3, 4, 5, 21 to 23)</summary>

```bash
# End-task estimators and the archived common-start reference
$PY scripts/rr_extrapolation.py --out-dir outputs/sweeps      # rr_extrapolation_rep{0,1,2}.jsonl and rr_extrapolation_gold.jsonl

# Dispersed-start reference: three start laws, same seed bases
for start in "zero 0.0" "chol 2.0" "gauss 1.0"; do set -- $start
  $PY scripts/rr_extrapolation.py --gold-only --start-mode $1 --start-scale $2 \
      --start-seed-base 345000000 --dynamics-seed-base 346000000 --attempt 0 \
      --gold-out outputs/sweeps/rr_extrapolation_gold_p4_fd354ef.jsonl
done
# The Adult cell at prior 0.3 was repeated at doubled length with `--attempt 1`,
# as the protocol allows; each row of the included file records its command.

$PY scripts/gold_rescore_p4.py --run --out outputs/sweeps/gold_rescore_p4.jsonl
$PY scripts/map_baseline.py
$PY scripts/vi_baseline.py --out outputs/sweeps/vi_baseline.jsonl      # CPU

for r in 0 1 2; do
  CUDA_VISIBLE_DEVICES=$r $PY scripts/variance_ratio_check.py --replicate $r --out outputs/sweeps/variance_ratio_rep$r.jsonl
done
$PY scripts/certified_gamma_table.py      # CPU; writes outputs/screens/certified_gamma_screen.jsonl
```
</details>

<details>
<summary>Control variates and the bimodal target (Tables 31, 32, 33)</summary>

The ZV and bimodal scripts stamp every row with the hash of a protocol file
and will not start a campaign without one. Pass any text file that describes
your run.

```bash
# Tables 31-32. Fit the control-variate coefficients, run, then summarize.
$PY scripts/zv_composition.py --selftest
$PY scripts/zv_composition.py --fit      --precommit MY_PROTOCOL.md      # writes outputs/zv_frozen_<cell>.json
$PY scripts/zv_composition.py --campaign --precommit MY_PROTOCOL.md --out outputs/zv_campaign.jsonl
$PY scripts/zv_composition.py --emit     --precommit MY_PROTOCOL.md --out outputs/zv_campaign.jsonl

# Table 33
$PY scripts/multimodal_bimodal.py --selftest
$PY scripts/multimodal_bimodal.py --campaign --precommit MY_PROTOCOL.md --out outputs/multimodal_campaign.jsonl
$PY scripts/multimodal_bimodal.py --emit     --precommit MY_PROTOCOL.md --out outputs/multimodal_campaign.jsonl
```

`--emit` checks the rows and writes the summary `<out>.emitter.json`, which
`render_study_tables.py` turns into a table.
</details>

## Notes

- Tables 2, 20 and 27 are included as printed and have no generator here.
  Table 2 is rounded from Tables 9 and 10.
- `chees_real_logistic.py` (Table 27) is included for reference. It depends
  on a protocol file, a lock manifest and tuner sidecars that are not in the
  repository.
- Host names in the rows are replaced by `redacted-hpc-node`.
- Comments cite protocol items such as `V3-5` or `DR-4`. They refer to the
  study protocols described in Appendix G of the paper.
- Reruns use the recorded seeds. Results on other hardware or library
  versions need not match to the last digit.

## Data

The experiments use public datasets, downloaded by the three
`scripts/download_*.py` scripts, which record source and license metadata
next to the files.

| Dataset | Source | License |
| --- | --- | --- |
| Breast Cancer Wisconsin (Diagnostic) | UCI, DOI `10.24432/C5DW2B` | CC BY 4.0 |
| Spambase | UCI, DOI `10.24432/C53G6X` | CC BY 4.0 |
| Adult | UCI, DOI `10.24432/C5XW20` | CC BY 4.0 |
| Gisette | UCI, DOI `10.24432/C5HP5B` | CC BY 4.0 |
| Superconductivity | UCI, DOI `10.24432/C53P47` | CC BY 4.0 |
| MNIST, digits 7 and 9 | LeCun, Cortes and Burges | CC BY-SA 3.0 |

All other data in this repository were generated by the scripts.

## License

The code and the generated data are released under the MIT License; see
`LICENSE`.

## Citation

```bibtex
@inproceedings{farell2026parallel,
  title     = {How Should Parallel {L}angevin Chains Share Noise?},
  author    = {Farell, Ryan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026},
  url       = {https://openreview.net/forum?id=dTLaD3Us8h}
}
```
