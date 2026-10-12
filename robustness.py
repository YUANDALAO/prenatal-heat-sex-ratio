"""Pre-specified robustness checks (ANALYSIS_PLAN.md) for
  (a) H1: male ~ GEST + POST, hot tertile
  (b) H5 finding: neonatal death ~ GEST + GxM + male + POST + PxM, hot tertile (focus: GEST)
  (c) same as (b) for infant death
Checks: Conley 100 km SEs (grid-cell centroids), sampling weights, non-movers, births <=60 months before interview,
survey x birth-year FE instead of country x year-month FE (+ country x calendar-month seasonality),
leave-one-country-out, pre-conception window (months -18..-10) entered jointly.
"""
import os
import numpy as np, pandas as pd, xarray as xr
from scipy import stats
from analysis_primary import demean, fe_ols, DATA, RES

ds = xr.open_zarr(os.path.join(DATA, 'power_monthly_anom.zarr'))
LAT, LON = ds.lat.values, ds.lon.values; NLON = len(LON)
ANOM = ds['T2M_anom'].values; T0 = pd.Timestamp(ds.time.values[0])


def window_mean(cell, cmc, a, b):
    li, lo = cell // NLON, cell % NLON
    base = (cmc - 1) - ((T0.year - 1900) * 12 + (T0.month - 1))
    idx = base[:, None] + np.arange(a, b + 1)[None, :]
    ok = (idx >= 0) & (idx < ANOM.shape[0])
    v = np.where(ok, ANOM[np.clip(idx, 0, ANOM.shape[0] - 1), li[:, None], lo[:, None]], np.nan)
    return np.where(ok.all(1), v.mean(1), np.nan)


def conley(df, y, xs, fes, km=100):
    d = df[list(dict.fromkeys([y] + xs + fes + ['cell']))].dropna()
    M = demean([y] + xs, fes, d); Y, X = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ Y; e = Y - X @ b
    cells, ci = np.unique(d.cell.to_numpy(), return_inverse=True)
    S = np.zeros((len(cells), X.shape[1])); np.add.at(S, ci, X * e[:, None])
    la = np.radians(LAT[cells // NLON]); lo = np.radians(LON[cells % NLON])
    meat = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(cells)):     # uniform kernel within km (haversine)
        dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la - la[i]) / 2) ** 2 + np.cos(la[i]) * np.cos(la) * np.sin((lo - lo[i]) / 2) ** 2))
        w = dist <= km
        meat += np.outer(S[i], S[w].sum(0))
    V = XtX_inv @ meat @ XtX_inv; se = np.sqrt(np.diag(V))
    return pd.DataFrame({'term': xs, 'b_pp': b * 100, 'se_pp': se * 100, 'lo_pp': (b - 1.96 * se) * 100,
                         'hi_pp': (b + 1.96 * se) * 100, 'p': 2 * stats.norm.sf(np.abs(b / se)), 'n': len(Y), 'clusters': len(cells)})


def demean_w(cols, groups, df, w, tol=1e-9, max_iter=300):
    """Weighted alternating projections: within-group weighted means removed (exact weighted FE regression)."""
    M = df[cols].to_numpy(dtype=np.float64).copy()
    codes = [pd.factorize(df[g].to_numpy())[0] for g in groups]
    wsum = [np.bincount(c, weights=w) for c in codes]
    for _ in range(max_iter):
        change = 0.0
        for c, ws in zip(codes, wsum):
            for j in range(M.shape[1]):
                m = np.bincount(c, weights=w * M[:, j]) / ws
                M[:, j] -= m[c]; change = max(change, np.abs(m).max())
        if change < tol: break
    return M


def weighted(df, y, xs, fes, wcol, cluster='clus'):
    d = df[list(dict.fromkeys([y] + xs + fes + [cluster, wcol]))].dropna()
    w = d[wcol].to_numpy(dtype=np.float64); w = w / w.mean(); sw = np.sqrt(w)
    M = demean_w([y] + xs, fes, d, w)
    Y, X = M[:, 0] * sw, M[:, 1:] * sw[:, None]
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ Y; e = Y - X @ b
    cl = pd.factorize(d[cluster].to_numpy())[0]; G = cl.max() + 1
    sc = np.zeros((G, X.shape[1])); np.add.at(sc, cl, X * e[:, None])
    V = XtX_inv @ (sc.T @ sc) @ XtX_inv * G / (G - 1); se = np.sqrt(np.diag(V))
    return pd.DataFrame({'term': xs, 'b_pp': b * 100, 'se_pp': se * 100, 'lo_pp': (b - 1.96 * se) * 100,
                         'hi_pp': (b + 1.96 * se) * 100, 'p': 2 * stats.norm.sf(np.abs(b / se)), 'n': len(Y), 'clusters': G})


def main():
    cols = ['survey', 'v001', 'v005', 'v104', 'cc', 'b3', 'age_m', 'b5', 'b7', 'male', 'primary', 'clim_tertile', 'cell',
            'T2M_anom_GEST', 'T2M_anom_POST']
    B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=cols)
    B = B[B.primary & (B.clim_tertile == 2)].copy()
    B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str)
    B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
    B['by'] = 1900 + (B.b3 - 1) // 12; B['bm'] = (B.b3 - 1) % 12 + 1
    B['svy_by'] = B.survey + '_' + B.by.astype(str); B['cc_bm'] = B.cc + '_' + B.bm.astype(str)
    B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float); B['imr'] = ((B.b5 == 0) & (B.b7 < 12)).astype(float)
    B['GxM'] = B.T2M_anom_GEST * B.male; B['PxM'] = B.T2M_anom_POST * B.male
    B['PRE'] = window_mean(B.cell.to_numpy(), B.b3.to_numpy().astype(int), -18, -10)
    B['nonmover'] = (B.v104 == 95) | ((B.v104 < 95) & (B.v104 >= B.age_m / 12))
    specs = {'H1 male': ('male', ['T2M_anom_GEST', 'T2M_anom_POST'], B),
             'H5 neonatal death': ('nnd', ['T2M_anom_GEST', 'GxM', 'male', 'T2M_anom_POST', 'PxM'], B),
             'H5 infant death': ('imr', ['T2M_anom_GEST', 'GxM', 'male', 'T2M_anom_POST', 'PxM'], B[B.age_m >= 12])}
    FE = ['clus', 'ctym']; R = []

    def add(r, outcome, check):
        r = r[r.term == 'T2M_anom_GEST'].copy(); r['outcome'] = outcome; r['check'] = check; R.append(r); print(r.round(4).to_string(index=False), flush=True)

    for name, (y, xs, D) in specs.items():
        add(fe_ols(D, y, xs, FE, 'clus')[0], name, '0 primary specification')
        add(conley(D, y, xs, FE), name, '1 Conley 100 km SE')
        add(weighted(D, y, xs, FE, 'v005'), name, '2 sampling weights (v005)')
        add(fe_ols(D[D.nonmover], y, xs, FE, 'clus')[0], name, '3 non-movers only')
        add(fe_ols(D[D.age_m <= 60], y, xs, FE, 'clus')[0], name, '4 births <=60 months before interview')
        add(fe_ols(D, y, xs, ['clus', 'svy_by', 'cc_bm'], 'clus')[0], name, '5 survey x birth-year + country x month FE')
        add(fe_ols(D, y, xs + ['PRE'], FE, 'clus')[0], name, '7 pre-conception window added')
        for c in sorted(D.cc.unique()):
            r = fe_ols(D[D.cc != c], y, xs, FE, 'clus')[0]; r = r[r.term == 'T2M_anom_GEST'].copy()
            r['outcome'] = name; r['check'] = f'6 leave out {c}'; R.append(r)
        L = pd.concat([r for r in R if r.outcome.iloc[0] == name and r.check.iloc[0].startswith('6')])
        print(name, 'leave-one-country-out GEST range:', round(L.b_pp.min(), 3), round(L.b_pp.max(), 3),
              '| max p:', round(L.p.max(), 4), flush=True)
    O = pd.concat(R, ignore_index=True); O.to_csv(os.path.join(RES, 'robustness.csv'), index=False)


if __name__ == '__main__':
    main()
