# Thermal stress in gestation and the sex ratio at birth — locked analysis plan

Locked 2026-10-08. This is after DHS access was approved and after the exposure-variance feasibility run
(synthetic births, no DHS data), but **before any DHS birth or outcome data were opened**.
Any later change is listed under Deviations, with its date and reason.

## Background that shaped the hypotheses (verified 2026-10-08)
- Nordic cohorts, 1878–1914: cold gestations lowered the share of male births and lengthened male life span.
  Catalano, Bruckner & Smith, PNAS 2008, PMID 18250336.
- Historical Sami: warm years brought more sons, about 1% more sons per 1 °C (≈ +0.25 pp male share).
  Helle et al., Biol Lett 2008, PMID 18042510.
- Finland 1865–2003: male excess in warm years. Helle et al., J Anim Ecol 2009, PMID 19719518.
- Poznań 1848–1900: warmer conception years, fewer girls. Liczbińska et al., Sci Rep 2024, PMID 38890431.

All evidence comes from cold-climate populations, where the thermal stressor is cold. The fetal-frailty
("culling") theory says the stressor, not temperature itself, removes frail male fetuses. It therefore predicts
the **opposite sign in hot climates**: heat should lower the male share. A temperature-per-se mechanism
predicts warmth raises the male share everywhere. The DHS spans hot lowlands and cool highlands, so the two
can be told apart.

## Data
- DHS Births Recode (BR) from all surveys with GPS; Women's Recode (IR) for the reproductive calendar and
  years lived in place; GE for cluster coordinates.
- NASA POWER MERRA-2 monthly temperature (0.5° × 0.625°), as anomalies from the cell's 1991–2020
  calendar-month mean (`power_monthly.py`, `exposure.py`).

## Sample
- **Included:** live births to interviewed women, born 1–180 months before interview (≤15 years), singletons
  (b0 = 0), births from 1982 onwards so the exposure window lies inside the climate record, and clusters with
  valid coordinates (excluding 0,0 and missing).
- **Primary-analysis exclusions:** India, China, Viet Nam, Armenia, Azerbaijan, Albania, Pakistan and Nepal
  (prenatal sex selection). Births with missing sex.
- **Secondary:** India and Nepal, first births only.
- **Sensitivity:** non-movers only (v104 ≥ child age, or "always" lived in place); births within 5 years of
  interview, where recall is best.

## Exposure
- **Primary.** GEST = mean T2M anomaly (°C) over the 9 calendar months before the birth month, at the
  cluster's grid cell.
- **Secondary.** Trimester means T1, T2, T3. T2M_MAX anomaly (monthly maximum). T2MWET anomaly (wet-bulb).
- **Baseline climate of the cell.** Mean annual T2M, 1991–2020. Tertiles define cool, temperate and hot
  climates (cut points fixed from the cluster-weighted distribution before outcomes are examined).

## Outcomes
- **Primary.** Male (b4 = 1) among live births.
- **Mechanism.** (a) Pregnancy termination in the IR calendar (DHS-6 to DHS-8). (b) Neonatal death
  (b7 = 0 among births ≥1 month before interview) by sex. (c) Infant death by sex.

## Models
Linear probability models with cluster fixed effects and country × birth-year-month fixed effects. Standard
errors are clustered by survey cluster. Unweighted primary analysis; weighted (v005) as sensitivity.

## Hypotheses (one primary test; the rest secondary or exploratory)

**H1 (primary, two-sided).** Heat anomaly during gestation changes the male share in hot climates.
Estimated in the hot tertile: male ~ GEST + POST + FE. POST is the same anomaly over birth..birth+11,
included as a negative control because anomalies are autocorrelated (r ≈ 0.38 in feasibility).
The culling theory predicts a negative coefficient on GEST, and a coefficient on POST near zero.

**H2 (secondary).** The sign differs by baseline climate: male ~ GEST × climate tertile. Culling predicts
negative in hot climates and positive in cool climates (where warmth relieves cold stress).
Test: GEST(hot) minus GEST(cool) < 0.

**H3 (secondary).** Non-linearity. Bins of GEST anomaly (<−1, −1 to −0.5, −0.5 to 0.5 [reference],
0.5 to 1, >1 °C) within each climate tertile.

**H4 (secondary, mechanism).** The heat anomaly over the 9 months before each calendar pregnancy outcome
raises the probability that the pregnancy ended in a termination. Same fixed-effects structure.

**H5 (secondary, culling prediction).** Among live births in hot climates, heat exposure lowers the
male-minus-female difference in neonatal mortality (interaction GEST × male).

**H6 (exploratory).** Trimester specificity (T1, T2, T3 entered jointly).

## Falsification and robustness
- POST and LEAD windows (H1).
- Pre-conception window (months −18 to −10) entered jointly.
- Conley spatial standard errors (100 km).
- Survey-specific trends in place of country × year-month fixed effects.
- Leave-one-country-out.
- ERA5-Land replication if a CDS account becomes available.

## Power (from `feasibility_exposure.py`)
- SD of GEST = 0.53 °C; 46% of its variance remains after fixed effects.
- With about 3.7 million births in total, the MDE is about 0.11 pp per SD, or 0.2 pp per °C.
- The hot tertile alone (about a third of births) gives an MDE of about 0.35 pp per °C.
- The Sami estimate (≈0.25 pp per °C) is below that. **H1 is therefore powered only for effects somewhat larger
  than those reported in cold climates.** A null result in H1 will be reported with its confidence interval as
  an upper bound on the effect, not as evidence of no effect.

## Reporting
All pre-specified estimates are reported whatever their direction, with code and derived (non-identifiable,
aggregated) results shared. DHS microdata and GPS coordinates are not redistributed.

## Deviations
These are data-handling decisions made on 2026-10-08, before any outcome was examined.

1. **Downloaded set.** 196 DHS surveys with GPS, out of the 210 flagged GPS in the DHS API.
   - 15 GPS datasets were not offered in the bulk list, probably because they are restricted:
     AF2015, AM2005, CM1998, ER2002, KE1998, MD2004, NI2006, PK2012, PE2010, PE2011, PE2012, RW2000,
     TZ1996, YE2013 and ZM2002.
   - They will be added if access is granted. Results will be reported with and without them.
2. **Peru continuous DHS 2003–08.** Survey IDs 228 (2004–06) and 447 (2007–08) are served by the identical
   file PEBR51FL / PEIR51FL (same MD5; 89,220 births, interview years 2003–2008).
   - It is used once, with the PEGE52FL coordinates.
   - The second copy is kept in dhs_raw/duplicates/ and not analysed.
3. **Senegal nested packages.** Some packages contained extra files not in the request (SN 6R and G0 BR/IR/GE).
   They are set aside in dhs_raw/extra_nested/ and not analysed.
5. **Completeness of the GPS set (verified against the DHS dataset catalog API, 2026-10-08).**
   - The catalog lists 195 standard-DHS GE files. All are in hand: PEGE5BFL (PE2007) contains the same
     1,414 clusters as PEGE52FL.
   - The 15 surveys in item 1 have **no GE file in the catalog at all**. GPS was collected but never released,
     so they cannot be obtained by application. Item 1 is corrected accordingly.
6. **Pre-declared replication sample (written while the primary analysis was running, before any result was
   viewed).** MIS and AIS surveys with released GPS: 44 surveys, mostly in sub-Saharan Africa.
   - Births are taken from BR where available (8 surveys). Otherwise they come from the IR women's file,
     which holds births in the last 5 years in wide format (36 surveys).
   - The sample is analysed separately as an independent replication of H1 and H2, with identical
     specification, and is **not pooled** with the primary DHS sample.
   - It is used only if the files can be obtained.
4. **National calendars.** Each survey's median interview CMC is compared with the API fieldwork midpoint.
   Surveys offset by ≥24 months are shifted by that offset; all offsets are logged in data/survey_qc.csv.
7. **Conley SEs (2026-10-08).** The analysis file stores no coordinates (privacy), so Conley 100 km SEs use the
   centroid of each 0.5° × 0.625° grid cell. Scores are summed within cells; the estimator was validated to equal
   cell-clustered SEs at zero distance.
8. **Exploratory stillbirth-sex analysis not possible.** DHS-8 records sex only for live births (p4 is missing for
   all stillbirths, miscarriages and abortions), so the sex of stillborn fetuses cannot be analysed.
9. **Replication sample definition (2026-10-08, after download and file checks, before reading any record).**
   All 96 MIS/AIS files were obtained (44 GE, 8 BR, 44 IR). The IR-only surveys hold births in the last
   5 years. For internal consistency, the whole replication sample uses births 1–60 months before interview,
   including the 8 BR surveys. All other rules are identical to the primary sample: singletons; births from 1982;
   valid coordinates; the same sex-selection exclusions; the same climate tertile cut points (from the primary
   sample); the same calendar check. The replication estimates are H1 (male share, hot tertile) and H5 (neonatal
   death, hot tertile, GEST main effect and GEST × male), with the primary specification.
10. **Replication sample composition.** 41 of 44 MIS/AIS GPS surveys contribute births. TZ2003AIS and MZ2009AIS
    have no birth history in IR and no BR. In UG2011AIS the birth-history variables are present but entirely empty;
    only v201 (children ever born) is filled. The sample has 215,772 births, 89,864 of them in the hot tertile.
11. **Replication of H5 is not feasible at useful power (2026-10-08).** In 33 of the 41 MIS/AIS surveys, age at
    death (b6 and b7) is empty for every dead child: MIS questionnaires record survival but not age at death.
    Neonatal death is therefore unmeasured there, and the all-survey estimate (NND rate 7.2‰) is invalid.
    Restricting to the 8 surveys with age at death (20,182 hot-tertile births; NND 31.9‰, matching DHS) gives
    GEST +0.14 pp/°C (−1.28 to +1.55). The point estimate is close to the primary +0.16, but power to detect
    +0.16 is 4%. The replication is reported as uninformative, not as a failure to replicate.
    H1 in all 41 surveys: −0.18 pp/°C (−1.66 to +1.31), also underpowered.

## Addendum C — conflict positive control (locked 2026-10-08, before any ACLED data were downloaded)

Purpose: show whether this design can detect stress-induced culling of male fetuses at all. If it can (conflict lowers
the sex ratio) while heat does not, heat's null is informative; if conflict also shows nothing, the null on heat
must be reported as possibly reflecting design insensitivity.

- Data: ACLED event-level data (acled_fetch.py), primary-analysis countries, all available years. Events with
  geo_precision 3 (only province known) or time_precision 3 (only month known) are excluded.
- Exposure (C-GEST): indicator that at least one event with >= 1 fatality occurred within 50 km of the DHS cluster
  during gestation months -9..-1 relative to birth. Negative control (C-POST): same indicator for months 0..11.
  Only births whose whole window -9..+11 lies inside the ACLED coverage of that country (first full year of coverage
  onward) are used.
- Model: male ~ C-GEST + C-POST | cluster + country x year-month; CR1 SEs by cluster; primary countries, all climate
  tertiles. Prediction (culled cohort): C-GEST < 0; C-POST ~ 0. One-sided interpretation not used; report 95% CI.
- Secondary: (a) dose: log(1 + fatalities within 50 km) during gestation; (b) 25 km radius; (c) neonatal death as
  outcome (scarring vs culling); (d) heat GEST added to the same model.
- Decision rule stated in advance: positive control "passes" if C-GEST is negative with p < 0.05 and C-POST is not;
  either way the result is reported. This is exploratory and not part of the original pre-registration.
- Result (2026-10-08, analysis_conflict.py, results/conflict_results.csv): 2 688 024 births, 42 countries, 27% exposed.
  C-GEST +0.05 pp (-0.16 to 0.26), C-POST -0.04 pp; dose, 25 km and neonatal-death variants all null.
  By the rule stated above the positive control does NOT pass. Conflict also did not raise neonatal death, so this
  exposure measure did not behave as a biological stressor in these data.

## Addendum D — major-war positive control, UCDP GED 26.1 (locked 2026-10-08, before UCDP data were downloaded; run once)

Reason: the ACLED control (Addendum C) did not raise neonatal death, so it was not a valid stressor. This single
replacement test uses only extreme organised violence and states a validity check in advance.
- Data: UCDP GED 26.1 (public; https://ucdp.uu.se/downloads/ged/ged261-csv.zip), all violence types, events with
  where_prec <= 3 and date_prec <= 3; deaths = `best`. Coverage 1989-2025 for every country.
- Exposure W-GEST: indicator that >= 1000 UCDP best-estimate deaths occurred within 100 km of the DHS cluster during
  gestation months -9..-1. W-POST: same for months 0..11 (negative control). Births with window -9..+11 outside
  Jan 1989 .. last GED month are dropped.
- Model: as Addendum C (male ~ W-GEST + W-POST | cluster + country x year-month; CR1 by cluster; primary countries).
- Validity check (stated first): W-GEST must raise neonatal death (b > 0, p < 0.05). If it does not, the control is
  declared invalid and the sex-ratio estimate is reported without interpretation as a sensitivity test.
- Pass rule: valid AND W-GEST lowers the male share (p < 0.05) AND W-POST does not.
- No further conflict definitions will be tried after this test, whatever its result.
- Result (2026-10-08, analysis_ucdp.py, results/ucdp_results.csv): 4 119 992 births, 56 countries; 38 696 births (0.94%,
  5004 clusters) exposed. Validity check FAILED: neonatal death W-GEST -0.15 pp (-0.45 to 0.15). Male share W-GEST
  -0.65 pp (-1.53 to 0.23), p = 0.15; W-POST +0.24. Declared invalid per the locked rule; reported without
  interpretation. Post hoc observation (not acted on): 35% of exposed births are Jordanian clusters within 100 km of
  Syrian war deaths across the border, so the 100 km radius mixes cross-border exposure. No further conflict
  definitions will be tried.
12. **Effect for all births (2026-10-12, after results were known).** In the hypothesis 5 model the anomaly is entered
    with an anomaly x male interaction, so its coefficient is the effect in girls (0.16 pp per degC). The headline
    mortality result and the illustrative burden now use the effect for all births, from the same model without the
    interaction and with sex as a covariate (analysis_h5_pooled.py): neonatal death 0.12 (0.01 to 0.23), infant death
    0.20 (0.04 to 0.35). The burden falls from 29 012 to 21 974 (2054 to 41 895) extra neonatal deaths a year. The
    estimate for girls from the pre-specified model is reported alongside.
