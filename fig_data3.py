"""Revised figure data.
(1) EXPLORATORY lag profile with 3-month blocks (single-month model gave sawtooth estimates from collinearity):
    blocks -18..-16, -15..-13, -12..-10 (pre-conception), -9..-7 (T1), -6..-4 (T2), -3..-1 (T3), 0..2, 3..5, 6..8, 9..11.
(2) EXPLORATORY country estimates for neonatal death using hot-tertile births only (matches the pre-specified H5 sample).
"""
import os, numpy as np, pandas as pd
from analysis_primary import fe_ols, DATA, RES
from robustness import window_mean

cols = ['survey', 'v001', 'cc', 'b3', 'male', 'primary', 'clim_tertile', 'cell', 'age_m', 'b5', 'b7', 'T2M_anom_GEST', 'T2M_anom_POST']
B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=cols); B = B[B.primary].copy()
B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str); B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float)
blocks = [(-18, -16), (-15, -13), (-12, -10), (-9, -7), (-6, -4), (-3, -1), (0, 2), (3, 5), (6, 8), (9, 11)]
cell = B.cell.to_numpy(); cmc = B.b3.to_numpy().astype(int)
names = []
for a, b in blocks:
    n = f'K{a}_{b}'; B[n] = window_mean(cell, cmc, a, b); names.append(n)
rows = []
for t, outc in [(2, 'male'), (0, 'male'), (2, 'nnd'), (0, 'nnd')]:
    r, _, _ = fe_ols(B[B.clim_tertile == t], outc, names, ['clus', 'ctym'], 'clus')
    r['mid'] = [(a + b) / 2 for a, b in blocks]; r['tertile'] = t; r['outcome'] = outc; rows.append(r); print('blocks', outc, t, flush=True)
pd.concat(rows).to_csv(os.path.join(RES, 'fig_lagblocks.csv'), index=False)
crow = []
for c, D in B[B.clim_tertile == 2].groupby('cc'):
    if len(D) < 15000: continue
    try:
        r, _, _ = fe_ols(D, 'nnd', ['T2M_anom_GEST', 'T2M_anom_POST'], ['clus', 'ctym'], 'clus')
        crow.append(dict(cc=c, outcome='nnd_hot', b=r.b_pp.iloc[0], se=r.se_pp.iloc[0], n=len(D), hot_share=1.0))
    except np.linalg.LinAlgError: pass
pd.DataFrame(crow).to_csv(os.path.join(RES, 'fig_country_nnd_hot.csv'), index=False); print('done', len(crow), 'countries')
