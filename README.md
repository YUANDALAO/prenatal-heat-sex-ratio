# Heat in pregnancy, the sex ratio at birth and neonatal survival

Analysis code and aggregate results for:

> Chen Y, et al. Heat in pregnancy, the sex ratio at birth, and neonatal survival in 56 low-income and middle-income countries: an analysis of 4·4 million births. (Manuscript under review.)

Births from 195 Demographic and Health Surveys (DHS) with cluster coordinates are linked to monthly NASA POWER (MERRA-2) temperature anomalies during pregnancy. Models use cluster and country × birth-month fixed effects, so each estimate compares warmer and cooler pregnancies in the same place.

## Analysis plan

`ANALYSIS_PLAN.md` was written and locked on 8 October 2026, before any DHS birth or outcome record was opened. It was not lodged in a public registry. Every later change is listed there with its date and reason (deviations 1–12). Two exploratory positive controls using conflict data were added later; each had its own short plan (addenda C and D), locked before those data were downloaded.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Climate | `power_monthly.py` | `data/power_monthly_anom.zarr` (public, anonymous AWS access) |
| Exposure linkage | `exposure.py` | function used by the build scripts |
| Births | `build_births.py` | `data/births.parquet`, `data/survey_qc.csv`, `data/climate_tertiles.json` |
| Pregnancies (hypothesis 4) | `build_pregnancies.py` | `data/pregnancies.parquet` |
| MIS/AIS replication sample | `build_births_mis.py` | `data/births_mis_ais.parquet` |
| Hypotheses 1–3, 6 | `analysis_primary.py` | `results/primary_results.csv` |
| Hypothesis 4 | `analysis_h4.py` | `results/h4_results.csv` |
| Hypothesis 5 | `analysis_h5.py` | `results/h5_results.csv` |
| Mortality effect for all births (main result, burden) | `analysis_h5_pooled.py` | `results/h5_pooled.csv` |
| Robustness | `robustness.py` | `results/robustness.csv` |
| Replication | `analysis_replication.py` | `results/replication_results.csv` |
| Exploratory: twinning, Trivers–Willard | `analysis_evo.py` | `results/evo_results.csv` |
| Exploratory: conflict controls | `acled_token.py`, `acled_fetch.py`, `analysis_conflict.py`, `analysis_ucdp.py` | `results/conflict_results.csv`, `results/ucdp_results.csv` |
| Figure data | `fig_data.py`, `fig_data2.py`, `fig_data3.py`, `country_table.py` | `results/fig_*.csv`, `results/country_*.csv` |
| Figures | `figs_lancet.py` and `figs_windows.py` (main; `fig_windows.py` estimates the exposure windows), `figs_dense.py`, `figs_heat.py` (appendix) | `figures/` |

`feasibility_exposure.py` is the pre-registration power check on synthetic births.

## Data

DHS microdata and cluster coordinates cannot be redistributed; see `DATA_ACCESS.md` for how to obtain them and where to put them. The `results/` folder holds only aggregate estimates. Two figure-data files are withheld because they are at the level of individual births or small grid cells (`fig_anom_sample.csv`, `fig_map_cells.csv`); `fig_data.py` and `fig_data2.py` rebuild them from the DHS files.

Included public metadata: DHS API survey lists (`data/dhs_api_surveys*.csv`, fetched 2026-09-30), survey-level quality control and calendar offsets (`data/survey_qc*.csv`), the fixed climate tertile cut points, and Natural Earth 1:110m country boundaries (public domain).

## Environment

Python 3.12; package versions in `requirements.txt`. The analysis file has 6·1 million births, so the fixed-effects models need a machine with ample memory.

## Licence

Code: MIT (see `LICENSE`). DHS, NASA POWER, UN WPP, ACLED and UCDP data remain under their providers' terms.
