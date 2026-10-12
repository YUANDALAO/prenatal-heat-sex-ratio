"""Both-sexes effect of gestational heat on neonatal and infant death (hot tertile and all tertiles).
The pre-specified hypothesis 5 model includes GEST x male, so its GEST coefficient is the effect in girls. The effect for
all births is estimated here with male as a covariate and no interaction (same fixed effects, POST as negative control).
This estimate is used for the headline result and the illustrative burden. Output: results/h5_pooled.csv
"""
import os, numpy as np, pandas as pd
from analysis_primary import fe_ols, DATA, RES

cols = ['survey', 'v001', 'cc', 'b3', 'male', 'primary', 'clim_tertile', 'age_m', 'b5', 'b7', 'T2M_anom_GEST', 'T2M_anom_POST']
B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=cols); B = B[B.primary].copy()
B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str); B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float); B['imr'] = ((B.b5 == 0) & (B.b7 < 12)).astype(float)
R = []
for out, minage in [('nnd', 1), ('imr', 12)]:
    for lab, D in [('hot', B[B.clim_tertile == 2]), ('all primary', B)]:
        D = D[D.age_m >= minage]
        r, _, _ = fe_ols(D, out, ['T2M_anom_GEST', 'male', 'T2M_anom_POST'], ['clus', 'ctym'], 'clus')
        r['model'] = f'{out} ({lab}), both sexes'; r['rate_per1000'] = D[out].mean() * 1000; R.append(r)
O = pd.concat(R, ignore_index=True); O.to_csv(os.path.join(RES, 'h5_pooled.csv'), index=False)
pd.set_option('display.width', 200); print(O[['model', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n', 'rate_per1000']].round(4).to_string(index=False))
