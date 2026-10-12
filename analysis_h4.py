"""H4 (pre-specified): heat anomaly in the 9 months before a pregnancy outcome and the probability that the pregnancy
ended in a termination (vs live birth). Unit: pregnancy outcome in the 60 months before interview.
FE: survey cluster + country x outcome-year-month; SEs clustered by survey cluster. Coefficients in pp per 1 degC.
Exploratory (not pre-specified, labelled as such): DHS-8 stillbirths only; sex of stillborn fetuses.
"""
import os
import numpy as np, pandas as pd
from analysis_primary import fe_ols, DATA, RES

E = pd.read_parquet(os.path.join(DATA, 'pregnancies.parquet'))
E['clus'] = E.survey + '_' + E.v001.astype(int).astype(str)
E['ctym'] = E.cc + '_' + E.cmc_end.astype(int).astype(str)
P = E[E.primary].copy()
X = ['T2M_anom_GEST', 'T2M_anom_POST']
R = []
for lab, D in [('H4 primary sample, all climates', P), ('H4 hot tertile', P[P.clim_tertile == 2]),
               ('H4 temperate tertile', P[P.clim_tertile == 1]), ('H4 cool tertile', P[P.clim_tertile == 0]),
               ('H4 sensitivity: calendar surveys (DHS-4..7)', P[P.source == 'calendar']),
               ('H4 sensitivity: DHS-8 pregnancy history', P[P.source == 'dhs8'])]:
    r, _, _ = fe_ols(D, 'term', X, ['clus', 'ctym'], 'clus')
    r['model'] = f'{lab}; termination rate {D.term.mean() * 100:.1f}%'; R.append(r)

# exploratory, DHS-8 only
D8 = P[P.source == 'dhs8'].copy()
S = D8[D8.p32.isin([1, 2])].copy(); S['still'] = (S.p32 == 2).astype(float)
r, _, _ = fe_ols(S, 'still', X, ['clus', 'ctym'], 'clus'); r['model'] = f'EXPLORATORY stillbirth vs live birth (DHS-8); rate {S.still.mean() * 1000:.1f}/1000'; R.append(r)
SB = D8[(D8.p32 == 2) & D8.sex.isin([1, 2])].copy(); SB['male'] = (SB.sex == 1).astype(float)
if len(SB) > 1000:
    r, _, _ = fe_ols(SB, 'male', ['T2M_anom_GEST'], ['ctym'], 'clus')
    r['model'] = f'EXPLORATORY male share among stillbirths (DHS-8, n={len(SB)}, male {SB.male.mean() * 100:.1f}%)'; R.append(r)

O = pd.concat(R, ignore_index=True); O.to_csv(os.path.join(RES, 'h4_results.csv'), index=False)
pd.set_option('display.width', 250)
print(O[['model', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n', 'clusters']].round(4).to_string(index=False))
