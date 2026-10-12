"""EXPLORATORY major-war positive control (ANALYSIS_PLAN.md Addendum D, locked 2026-10-08 before download; run once).
W-GEST = >= 1000 UCDP GED 26.1 best-estimate deaths within 100 km of the cluster in months -9..-1; W-POST months 0..11.
Validity check first: W-GEST must raise neonatal death. Pass: valid AND W-GEST lowers the male share AND W-POST does not.
"""
import os, zipfile
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
import build_births as bb
from analysis_primary import fe_ols, DATA, RES

EARTH = 6371.0; R_KM = 100; THRESH = 1000
with zipfile.ZipFile(os.path.join(DATA, 'ucdp', 'ged261-csv.zip')) as z:
    E = pd.read_csv(z.open('GEDEvent_v26_1.csv'), usecols=['date_start', 'latitude', 'longitude', 'best', 'where_prec', 'date_prec'],
                    parse_dates=['date_start'])
E = E[(E.where_prec <= 3) & (E.date_prec <= 3) & (E.best > 0)].copy()
E['cmc'] = (E.date_start.dt.year - 1900) * 12 + E.date_start.dt.month
t0 = (1989 - 1900) * 12 + 1; t1 = int(E.cmc.max()); nT = t1 - t0 + 1
print('GED events', len(E), 'deaths', int(E.best.sum()), 'months', nT, flush=True)

q = pd.read_csv(os.path.join(DATA, 'survey_qc.csv')); q = q[q.status == 'ok']
G = pd.concat([bb.read_ge(r.ge).assign(survey=r.file[:8]) for r in q.itertuples()], ignore_index=True)
B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=['survey', 'v001', 'cc', 'b3', 'male', 'primary', 'b5', 'b7', 'T2M_anom_GEST', 'T2M_anom_POST'])
B = B[B.primary & (B.b3 - 9 >= t0) & (B.b3 + 11 <= t1)].copy(); B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float)
cl = B[['survey', 'v001']].drop_duplicates().merge(G, on=['survey', 'v001']).reset_index(drop=True)

tree = BallTree(np.radians(E[['latitude', 'longitude']].to_numpy()), metric='haversine')
idx = tree.query_radius(np.radians(cl[['lat', 'lon']].to_numpy()), r=R_KM / EARTH)
A = np.zeros((len(cl), nT + 1), dtype=np.float32); em = E.cmc.to_numpy() - t0; eb = E.best.to_numpy()
for i, ii in enumerate(idx):
    if len(ii): np.add.at(A[i], em[ii] + 1, eb[ii])
A = A.cumsum(1)
B = B.merge(cl[['survey', 'v001']].reset_index(), on=['survey', 'v001'], how='inner')
k = B['index'].to_numpy(); m = B.b3.to_numpy().astype(int) - t0
win = lambda a, c: A[k, m + c + 1] - A[k, m + a]
B['deaths_GEST'] = win(-9, -1); B['deaths_POST'] = win(0, 11)
B['W_GEST'] = (B.deaths_GEST >= THRESH).astype(float); B['W_POST'] = (B.deaths_POST >= THRESH).astype(float)
B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str); B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
ex = B[B.W_GEST == 1]
print('births', len(B), 'countries', B.cc.nunique(), 'exposed W_GEST', len(ex), round(B.W_GEST.mean(), 4),
      'exposed clusters', ex.clus.nunique(), flush=True)
print('exposed births by country / birth year:\n', ex.assign(yr=1900 + (ex.b3 - 1) // 12).groupby('cc').yr.agg(['size', 'min', 'max']).sort_values('size', ascending=False).head(20))

FE = ['clus', 'ctym']; rows = []
for name, y, xs in [('VALIDITY: neonatal death', 'nnd', ['W_GEST', 'W_POST']),
                    ('PRIMARY: male share', 'male', ['W_GEST', 'W_POST']),
                    ('male share, heat added', 'male', ['W_GEST', 'W_POST', 'T2M_anom_GEST', 'T2M_anom_POST'])]:
    r, _, _ = fe_ols(B, y, xs, FE, 'clus'); r['model'] = name; rows.append(r)
O = pd.concat(rows, ignore_index=True); O.to_csv(os.path.join(RES, 'ucdp_results.csv'), index=False)
pd.set_option('display.width', 250)
print(O[['model', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n', 'clusters']].round(4).to_string(index=False))
v = O[(O.model.str.startswith('VALID')) & (O.term == 'W_GEST')].iloc[0]; s = O[(O.model.str.startswith('PRIMARY'))].set_index('term')
valid = v.b_pp > 0 and v.p < 0.05
passed = valid and s.loc['W_GEST', 'b_pp'] < 0 and s.loc['W_GEST', 'p'] < 0.05 and s.loc['W_POST', 'p'] >= 0.05
print('VALID control:', valid, '| PASS:', passed)
