"""H5 (pre-specified): culling prediction on sex-specific early mortality.
Neonatal death = died in month 0 (b5 == 0 & b7 == 0), among births >= 1 month before interview.
Infant death  = died before month 12, among births >= 12 months before interview.
Model: death ~ GEST + GEST x male + male + POST + POST x male, cluster FE + country x birth-year-month FE,
SEs clustered by survey cluster. Culling predicts GEST x male < 0 in the hot tertile.
"""
import os
import numpy as np, pandas as pd
from analysis_primary import fe_ols, DATA, RES

cols = ['survey', 'v001', 'cc', 'b3', 'male', 'primary', 'clim_tertile', 'age_m', 'b5', 'b7',
        'T2M_anom_GEST', 'T2M_anom_POST']
B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=cols)
B = B[B.primary].copy()
B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str)
B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float)
B['imr'] = ((B.b5 == 0) & (B.b7 < 12)).astype(float)
B['GxM'] = B.T2M_anom_GEST * B.male; B['PxM'] = B.T2M_anom_POST * B.male
R = []
for out, minage in [('nnd', 1), ('imr', 12)]:
    for lab, D in [('hot', B[B.clim_tertile == 2]), ('all primary', B)]:
        D = D[D.age_m >= minage]
        r, _, _ = fe_ols(D, out, ['T2M_anom_GEST', 'GxM', 'male', 'T2M_anom_POST', 'PxM'], ['clus', 'ctym'], 'clus')
        r['model'] = f'H5 {out} ({lab}); mean rate {D[out].mean() * 1000:.1f}/1000'; R.append(r)
O = pd.concat(R, ignore_index=True); O.to_csv(os.path.join(RES, 'h5_results.csv'), index=False)
pd.set_option('display.width', 250)
print(O[['model', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n']].round(4).to_string(index=False))
