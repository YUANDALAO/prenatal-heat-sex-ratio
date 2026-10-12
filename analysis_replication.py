"""Pre-declared replication (ANALYSIS_PLAN.md deviation 9) in the independent MIS/AIS sample.
Identical specifications to the primary analyses: cluster FE + country x birth-year-month FE, SEs clustered by
survey cluster, hot climate tertile (primary-sample cut points). For comparison, the primary DHS estimates are
re-run on the same birth window (1-60 months before interview).
"""
import os
import numpy as np, pandas as pd
from analysis_primary import fe_ols, DATA, RES


def prep(B):
    B = B[B.primary & (B.clim_tertile == 2)].copy()
    B['clus'] = B.survey.astype(str) + '_' + B.surv_id.astype(str) + '_' + B.v001.astype(int).astype(str)
    B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
    B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float)
    B['GxM'] = B.T2M_anom_GEST * B.male; B['PxM'] = B.T2M_anom_POST * B.male
    return B


R = []
cols = ['survey', 'surv_id', 'v001', 'cc', 'b3', 'age_m', 'b5', 'b7', 'male', 'primary', 'clim_tertile',
        'T2M_anom_GEST', 'T2M_anom_POST']
for lab, path, window in [('REPLICATION MIS/AIS (births 1-60 m)', 'births_mis_ais.parquet', 60),
                          ('PRIMARY DHS, same window (births 1-60 m)', 'births.parquet', 60)]:
    B = prep(pd.read_parquet(os.path.join(DATA, path), columns=cols))
    B = B[B.age_m <= window]
    r, _, _ = fe_ols(B, 'male', ['T2M_anom_GEST', 'T2M_anom_POST'], ['clus', 'ctym'], 'clus'); r['model'] = f'{lab}: H1 male'; R.append(r)
    r, _, _ = fe_ols(B, 'nnd', ['T2M_anom_GEST', 'GxM', 'male', 'T2M_anom_POST', 'PxM'], ['clus', 'ctym'], 'clus')
    r['model'] = f'{lab}: H5 neonatal death (rate {B.nnd.mean() * 1000:.1f}/1000)'; R.append(r)
O = pd.concat(R, ignore_index=True); O.to_csv(os.path.join(RES, 'replication_results.csv'), index=False)
pd.set_option('display.width', 250)
print(O[['model', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n', 'clusters']].round(4).to_string(index=False))
