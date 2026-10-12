"""Data for the dense multi-panel figures (descriptive / exploratory; labelled as such in legends).
1. Binned scatter: outcome and GEST both residualised on cluster + country x year-month FE, 20 quantile bins.
2. Heat map: residual outcome by baseline-climate bin x anomaly bin.
3. Monthly lag profile (EXPLORATORY): outcome on 30 single-month anomalies, months -18..+11 relative to birth.
4. Country-specific GEST coefficients (EXPLORATORY) for funnel plots.
5. Within-cluster share of GEST variance by tertile.
"""
import os, numpy as np, pandas as pd
from scipy import stats
from analysis_primary import demean, fe_ols, DATA, RES
from robustness import window_mean

cols = ['survey', 'v001', 'cc', 'b3', 'male', 'primary', 'clim_tertile', 'clim_mean', 'cell', 'age_m', 'b5', 'b7',
        'T2M_anom_GEST', 'T2M_anom_POST']
B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=cols); B = B[B.primary].copy()
B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str); B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float)
FE = ['clus', 'ctym']; out = {}

# 1 + 2: residualise once per tertile
binrows, heat = [], []
for t in range(3):
    D = B[B.clim_tertile == t].copy()
    M = demean(['male', 'nnd', 'T2M_anom_GEST'], FE, D)
    D['r_male'], D['r_nnd'], D['r_g'] = M[:, 0], M[:, 1], M[:, 2]
    D['q'] = pd.qcut(D.r_g, 20, labels=False, duplicates='drop')
    for sex in ['all', 'girls', 'boys']:
        S = D if sex == 'all' else D[D.male == (sex == 'boys')]
        g = S.groupby('q').agg(x=('r_g', 'mean'), m=('r_male', 'mean'), m_se=('r_male', 'sem'), d=('r_nnd', 'mean'),
                               d_se=('r_nnd', 'sem'), n=('r_g', 'size')).reset_index()
        g['tertile'] = t; g['sex'] = sex; binrows.append(g)
    heat.append(D[['clim_mean', 'r_g', 'r_male', 'r_nnd', 'T2M_anom_GEST']].assign(tertile=t))
pd.concat(binrows).to_csv(os.path.join(RES, 'fig2_binscatter.csv'), index=False)
H = pd.concat(heat)
H['cb'] = pd.cut(H.clim_mean, [-50, 16, 19, 21, 23, 24.5, 26, 27, 50],
                 labels=['<16', '16–19', '19–21', '21–23', '23–24.5', '24.5–26', '26–27', '>27'])
H['ab'] = pd.cut(H.T2M_anom_GEST, [-9, -0.75, -0.25, 0.25, 0.75, 9], labels=['< −0.75', '−0.75 to −0.25', '±0.25', '0.25 to 0.75', '> 0.75'])
hm = H.groupby(['cb', 'ab'], observed=True).agg(male=('r_male', 'mean'), nnd=('r_nnd', 'mean'), n=('r_male', 'size')).reset_index()
hm.to_csv(os.path.join(RES, 'fig_heatmap.csv'), index=False)

# 3: monthly lag profile (hot and cool tertiles)
lags = list(range(-18, 12))
cell = B.cell.to_numpy(); cmc = B.b3.to_numpy().astype(int)
for k in lags: B[f'L{k}'] = window_mean(cell, cmc, k, k)
lagrows = []
for t, outc in [(2, 'male'), (0, 'male'), (2, 'nnd'), (0, 'nnd')]:
    D = B[B.clim_tertile == t]
    xs = [f'L{k}' for k in lags]
    r, V, b = fe_ols(D, outc, xs, FE, 'clus')
    r['lag'] = lags; r['tertile'] = t; r['outcome'] = outc; lagrows.append(r)
    print('lag profile', outc, t, 'done', flush=True)
pd.concat(lagrows).to_csv(os.path.join(RES, 'fig_lagprofile.csv'), index=False)

# 4: country-specific GEST coefficients (all tertiles, primary sample; same FE)
crow = []
for c, D in B.groupby('cc'):
    if len(D) < 20000: continue
    for outc in ['male', 'nnd']:
        try:
            r, _, _ = fe_ols(D, outc, ['T2M_anom_GEST', 'T2M_anom_POST'], FE, 'clus')
            g = r.iloc[0]; crow.append(dict(cc=c, outcome=outc, b=g.b_pp, se=g.se_pp, n=len(D), hot_share=(D.clim_tertile == 2).mean()))
        except np.linalg.LinAlgError: pass
pd.DataFrame(crow).to_csv(os.path.join(RES, 'fig_country.csv'), index=False)

# 5: within-cluster variance share of GEST and raw anomaly distribution by tertile
vs = []
for t in range(3):
    D = B[B.clim_tertile == t]
    w = D.T2M_anom_GEST - D.groupby('clus').T2M_anom_GEST.transform('mean')
    vs.append(dict(tertile=t, sd=D.T2M_anom_GEST.std(), within_share=w.var() / D.T2M_anom_GEST.var()))
pd.DataFrame(vs).to_csv(os.path.join(RES, 'fig_variance.csv'), index=False)
B[['clim_tertile', 'T2M_anom_GEST']].sample(400000, random_state=1).to_csv(os.path.join(RES, 'fig_anom_sample.csv'), index=False)
print('done')
