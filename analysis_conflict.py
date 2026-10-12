"""EXPLORATORY positive control (ANALYSIS_PLAN.md Addendum C, locked 2026-10-08 before ACLED download).
Does armed-conflict exposure in gestation lower the sex ratio at birth (culled cohort), while post-natal exposure does not?
Exposure: fatal ACLED events (fatalities >= 1; geo_precision < 3; time_precision < 3) within R km of the DHS cluster,
summed over months -9..-1 (GEST) and 0..11 (POST) relative to the birth month. Births whose window -9..+11 is outside
the country's ACLED coverage (Jan of first year .. last event month) are dropped.
"""
import os, zipfile
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
import build_births as bb
from analysis_primary import fe_ols, DATA, RES

ACL = os.path.join(DATA, 'acled'); EARTH = 6371.0
T = pd.read_csv(os.path.join(RES, 'country_table.csv'))[['cc', 'iso3']]
q = pd.read_csv(os.path.join(DATA, 'survey_qc.csv')); q = q[q.status == 'ok']
# cluster coordinates from the GE files (not stored in the analysis file)
G = pd.concat([bb.read_ge(r.ge).assign(survey=r.file[:8]) for r in q.itertuples()], ignore_index=True)
B = pd.read_parquet(os.path.join(DATA, 'births.parquet'),
                    columns=['survey', 'v001', 'cc', 'b3', 'male', 'primary', 'b5', 'b7', 'age_m', 'clim_tertile', 'T2M_anom_GEST', 'T2M_anom_POST'])
B = B[B.primary].copy(); B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float)
cmc = lambda d: (d.dt.year - 1900) * 12 + d.dt.month

parts = []
for cc, iso3 in T.itertuples(index=False):
    f = os.path.join(ACL, iso3 + '.csv')
    if not os.path.exists(f): print('no ACLED file', iso3); continue
    E = pd.read_csv(f, parse_dates=['event_date'])
    cov0 = (E.year.min() - 1900) * 12 + 1; cov1 = int(cmc(E.event_date).max())
    E = E[(E.fatalities >= 1) & (E.geo_precision < 3) & (E.time_precision < 3)].copy(); E['cmc'] = cmc(E.event_date)
    b = B[B.cc == cc]
    b = b[(b.b3 - 9 >= cov0) & (b.b3 + 11 <= cov1)].copy()
    if b.empty or E.empty: print(iso3, 'no births in coverage'); continue
    cl = b[['survey', 'v001']].drop_duplicates().merge(G, on=['survey', 'v001'])
    tree = BallTree(np.radians(E[['latitude', 'longitude']].to_numpy()), metric='haversine')
    t0, t1 = cov0, cov1; nT = t1 - t0 + 1
    for R in [50, 25]:
        idx = tree.query_radius(np.radians(cl[['lat', 'lon']].to_numpy()), r=R / EARTH)
        ev = np.zeros((len(cl), nT + 1)); fa = np.zeros((len(cl), nT + 1))   # cumulative sums, column 0 = 0
        for i, ii in enumerate(idx):
            if len(ii):
                m = E.cmc.to_numpy()[ii] - t0
                np.add.at(ev[i], m + 1, 1); np.add.at(fa[i], m + 1, E.fatalities.to_numpy()[ii])
        ev = ev.cumsum(1); fa = fa.cumsum(1)
        k = b.merge(cl[['survey', 'v001']].reset_index(), on=['survey', 'v001'], how='left')['index'].to_numpy()
        ok = ~np.isnan(k.astype(float)); k = np.nan_to_num(k).astype(int); m = b.b3.to_numpy().astype(int) - t0
        def win(A, a, c): return A[k, m + c + 1] - A[k, m + a]
        b[f'ev{R}_GEST'] = np.where(ok, win(ev, -9, -1), np.nan); b[f'ev{R}_POST'] = np.where(ok, win(ev, 0, 11), np.nan)
        b[f'fa{R}_GEST'] = np.where(ok, win(fa, -9, -1), np.nan); b[f'fa{R}_POST'] = np.where(ok, win(fa, 0, 11), np.nan)
    parts.append(b); print(iso3, 'births', len(b), 'exposed50', round((b.ev50_GEST > 0).mean(), 3), flush=True)

D = pd.concat(parts, ignore_index=True).dropna(subset=['ev50_GEST'])
D['clus'] = D.survey + '_' + D.v001.astype(int).astype(str); D['ctym'] = D.cc + '_' + D.b3.astype(int).astype(str)
for R in [50, 25]:
    D[f'C{R}_GEST'] = (D[f'ev{R}_GEST'] > 0).astype(float); D[f'C{R}_POST'] = (D[f'ev{R}_POST'] > 0).astype(float)
    D[f'L{R}_GEST'] = np.log1p(D[f'fa{R}_GEST']); D[f'L{R}_POST'] = np.log1p(D[f'fa{R}_POST'])
D.drop(columns=['b5', 'b7']).to_parquet(os.path.join(DATA, 'births_conflict.parquet'), index=False)
FE = ['clus', 'ctym']; rows = []
specs = [('primary: any fatal event 50 km', 'male', ['C50_GEST', 'C50_POST']),
         ('dose: log(1+fatalities) 50 km', 'male', ['L50_GEST', 'L50_POST']),
         ('radius 25 km', 'male', ['C25_GEST', 'C25_POST']),
         ('neonatal death, 50 km', 'nnd', ['C50_GEST', 'C50_POST']),
         ('with heat GEST', 'male', ['C50_GEST', 'C50_POST', 'T2M_anom_GEST', 'T2M_anom_POST'])]
for name, y, xs in specs:
    S = D            # all births have age >= 1 month, so neonatal status is complete (as in H5)
    r, _, _ = fe_ols(S, y, xs, FE, 'clus'); r['model'] = name; rows.append(r); print(name, flush=True)
O = pd.concat(rows, ignore_index=True); O.to_csv(os.path.join(RES, 'conflict_results.csv'), index=False)
pd.set_option('display.width', 250)
print('births', len(D), 'countries', D.cc.nunique(), 'exposed share (50 km GEST)', round(D.C50_GEST.mean(), 3))
print(O[['model', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n', 'clusters']].round(4).to_string(index=False))
