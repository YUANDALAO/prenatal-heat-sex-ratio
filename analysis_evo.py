"""EXPLORATORY (post hoc, not in ANALYSIS_PLAN.md): two evolutionary tests added after the primary results.

E1 Twinning. Stress lowers twinning (maternal-condition index; Bruckner & Catalano). Unit = delivery
   (b0 in {0,1}); outcome twin = b0 == 1. Same sample rules as the primary file except multiples are kept.
E2 Trivers-Willard by maternal condition. Low-condition mothers exposed to heat should bear fewer sons.
   Condition markers from the woman's record: short stature (height < 150 cm, v438), low BMI (< 18.5, v445),
   poorest two wealth quintiles (v190). Model: male ~ GEST + GEST x low + low + POST | cluster + country x year-month.

Exposure: GEST (months -9..-1) and POST (0..11) anomalies from the same grid cell as the primary file
(cluster -> cell map taken from data/births.parquet). Primary (non-sex-selecting) countries only.
"""
import os, re, io, zipfile
import numpy as np, pandas as pd
import build_births as bb
from analysis_primary import fe_ols, DATA, RES
from robustness import window_mean

COLS = ['v001', 'v002', 'v003', 'v008', 'bidx', 'b0', 'b3', 'b4', 'v190', 'v437', 'v438', 'v445']
CACHE = os.path.join(DATA, 'births_evo.parquet')


def read_cols(zname, cols):
    with zipfile.ZipFile(os.path.join(bb.RAW, zname)) as z:
        raw = z.read([i for i in z.namelist() if i.lower().endswith('.dta')][0])
    have = pd.read_stata(io.BytesIO(raw), iterator=True, convert_categoricals=False).variable_labels()
    return pd.read_stata(io.BytesIO(raw), columns=[c for c in cols if c in have], convert_categoricals=False)


def build():
    q = pd.read_csv(os.path.join(DATA, 'survey_qc.csv')); q = q[q.status == 'ok'].set_index('file')
    parts = []
    for f, r in q.iterrows():
        if f[:2] in bb.SEX_SELECTION: continue
        b = read_cols(f, COLS); s = int(r.cmc_shift_applied)
        b['b3'] += s; b['v008'] += s; b['age_m'] = b.v008 - b.b3
        b = b[(b.age_m >= 1) & (b.age_m <= 180) & (b.b3 >= (1982 - 1900) * 12 + 1) & b.b4.isin([1, 2]) & b.b0.isin([0, 1, 2, 3])]
        b = b.assign(survey=f[:8], cc=f[:2]); parts.append(b); print(f, len(b), flush=True)
    B = pd.concat(parts, ignore_index=True)
    for c in ['v190', 'v437', 'v438', 'v445']:
        if c not in B: B[c] = np.nan
    B.to_parquet(CACHE, index=False)


if not os.path.exists(CACHE): build()
B = pd.read_parquet(CACHE)
P = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=['survey', 'v001', 'cell', 'clim_tertile', 'primary'])
cm = P[P.primary].drop_duplicates(['survey', 'v001'])[['survey', 'v001', 'cell', 'clim_tertile']]
B = B.merge(cm, on=['survey', 'v001'], how='inner')
cell = B.cell.to_numpy(); c = B.b3.to_numpy().astype(int)
B['GEST'] = window_mean(cell, c, -9, -1); B['POST'] = window_mean(cell, c, 0, 11)
B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str); B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
FE = ['clus', 'ctym']; X = ['GEST', 'POST']; rows = []
TN = {0: 'cool', 1: 'temperate', 2: 'hot', 9: 'all'}


def add(r, test, sample, extra=None):
    r = r.copy(); r['test'] = test; r['sample'] = sample
    for k, v in (extra or {}).items(): r[k] = v
    rows.append(r)


# E1 twinning: one row per delivery
D = B[B.b0.isin([0, 1])].copy(); D['twin'] = (D.b0 == 1).astype(float)
print('deliveries', len(D), 'twin rate per 1000', round(D.twin.mean() * 1000, 2), flush=True)
for t in [2, 1, 0, 9]:
    S = D if t == 9 else D[D.clim_tertile == t]
    r, _, _ = fe_ols(S, 'twin', X, FE, 'clus'); add(r, 'E1 twin delivery', TN[t], {'base_per1000': S.twin.mean() * 1000})
    print('E1', TN[t], 'done', flush=True)

# E2 Trivers-Willard: singletons
S0 = B[B.b0 == 0].copy(); S0['male'] = (S0.b4 == 1).astype(float)
h = S0.v438.where(S0.v438.between(1000, 2200)) / 10; bmi = S0.v445.where(S0.v445.between(1200, 6000)) / 100
markers = {'short (<150 cm)': h.lt(150).where(h.notna()), 'low BMI (<18.5)': bmi.lt(18.5).where(bmi.notna()),
           'poorest 40%': S0.v190.isin([1, 2]).where(S0.v190.between(1, 5))}
for name, low in markers.items():
    S = S0.assign(low=low.astype(float)).dropna(subset=['low'])
    S['GxL'] = S.GEST * S.low
    for t in [2, 9]:
        T = S if t == 9 else S[S.clim_tertile == t]
        r, _, _ = fe_ols(T, 'male', ['GEST', 'GxL', 'low', 'POST'], FE, 'clus')
        add(r, f'E2 male x {name}', TN[t], {'share_low': T.low.mean(), 'male_pct_low': T.male[T.low == 1].mean() * 100,
                                         'male_pct_high': T.male[T.low == 0].mean() * 100})
        print('E2', name, TN[t], 'n', len(T), flush=True)

O = pd.concat(rows, ignore_index=True); O.to_csv(os.path.join(RES, 'evo_results.csv'), index=False)
pd.set_option('display.width', 250)
print(O[['test', 'sample', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n']].round(4).to_string(index=False))
print(O.groupby(['test', 'sample']).first()[[c for c in ['base_per1000', 'share_low', 'male_pct_low', 'male_pct_high'] if c in O]].round(3))
