"""Figure 2 data: exposure-window profile, hot tertile (the pre-specified sample).
Pre-conception (months -18..-10), pregnancy (GEST, -9..-1) and first year after birth (POST, 0..11) anomalies entered
jointly, with the primary fixed effects. Mortality models adjust for sex (both sexes pooled; sex-specific estimates are in
table 2). The pre-conception window is pre-specified robustness check 7; POST is the pre-specified negative control.
"""
import os, numpy as np, pandas as pd
from analysis_primary import fe_ols, DATA, RES
from robustness import window_mean

cols = ['survey', 'v001', 'cc', 'b3', 'male', 'primary', 'clim_tertile', 'cell', 'age_m', 'b5', 'b7', 'T2M_anom_GEST', 'T2M_anom_POST']
B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=cols)
B = B[B.primary & (B.clim_tertile == 2)].copy()
B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str); B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float); B['imr'] = ((B.b5 == 0) & (B.b7 < 12)).astype(float)
B['PRE'] = window_mean(B.cell.to_numpy(), B.b3.to_numpy().astype(int), -18, -10)
X = ['PRE', 'T2M_anom_GEST', 'T2M_anom_POST']; FE = ['clus', 'ctym']; rows = []
for y, extra, D in [('male', [], B), ('nnd', ['male'], B), ('imr', ['male'], B[B.age_m >= 12])]:
    r, _, _ = fe_ols(D, y, X + extra, FE, 'clus')
    r = r[r.term.isin(X)].assign(outcome=y, base=D[y].mean()); rows.append(r); print(y, 'done', flush=True)
O = pd.concat(rows, ignore_index=True); O.to_csv(os.path.join(RES, 'fig_windows.csv'), index=False)
pd.set_option('display.width', 200); print(O[['outcome', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n', 'base']].round(4).to_string(index=False))
