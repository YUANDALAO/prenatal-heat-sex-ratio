"""Exposure-variance feasibility (no DHS data needed): simulate births at random locations in ten large DHS
countries (bounding boxes), 2000-2020, and measure how much gestational temperature anomaly variance survives
cell and country x year-month fixed effects. This residual share feeds the power calculation."""
import numpy as np, pandas as pd
from exposure import link

rng = np.random.default_rng(2026)
BOX = {'Nigeria': (4.5, 13.5, 3, 14.5), 'Ethiopia': (4, 14.5, 34, 47), 'Kenya': (-4.5, 4.5, 34, 41.5),
       'DRC': (-13, 5, 13, 31), 'Tanzania': (-11.5, -1, 29.5, 40), 'Uganda': (-1.4, 4.2, 29.6, 35),
       'Ghana': (4.8, 11, -3.2, 1.2), 'Malawi': (-17, -9.4, 32.7, 35.9), 'Zambia': (-18, -8.2, 22, 33.7),
       'Mali': (10.2, 25, -12, 4.2)}
rows = []
for c, (la0, la1, lo0, lo1) in BOX.items():
    locs = np.c_[rng.uniform(la0, la1, 400), rng.uniform(lo0, lo1, 400)]          # 400 'clusters' per country
    k = rng.integers(0, 400, 20000)
    cmc = rng.integers((2000 - 1900) * 12 + 1, (2020 - 1900) * 12 + 13, 20000)
    rows.append(pd.DataFrame({'country': c, 'loc': [f'{c}_{i}' for i in k], 'lat': locs[k, 0], 'lon': locs[k, 1],
                              'cmc_birth': cmc}))
B = pd.concat(rows, ignore_index=True)
E = link(B); B = B.join(E)
B['ym'] = B.country + '_' + B.cmc_birth.astype(str)


def resid(y, groups, iters=30):
    r = y.copy()
    for _ in range(iters):
        for g in groups: r = r - r.groupby(B[g]).transform('mean')
    return r


out = []
for v in ['T2M_anom_GEST', 'T2M_anom_T1', 'T2M_anom_T3', 'T2M_MAX_anom_GEST', 'T2MWET_anom_GEST', 'T2M_anom_POST']:
    y = B[v]; r = resid(y - y.mean(), ['loc', 'ym'])
    out.append(dict(exposure=v, sd_raw=y.std(), sd_resid=r.std(), resid_var_share=r.var() / y.var(), n=y.notna().sum()))
O = pd.DataFrame(out)
print(O.round(3).to_string(index=False))
print('corr GEST vs POST (placebo overlap):', round(B.T2M_anom_GEST.corr(B.T2M_anom_POST), 3))
print('corr T1 vs T3:', round(B.T2M_anom_T1.corr(B.T2M_anom_T3), 3))
O.to_csv('data/feasibility_exposure.csv', index=False)
