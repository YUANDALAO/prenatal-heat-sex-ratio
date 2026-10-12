"""Country-level table for Lancet-style figures.
Per country: births, share of births in hot cells, baseline neonatal mortality (hot cells), sex-ratio estimate
(all births in the country; EXPLORATORY country stratification of H1/H2) and neonatal-death estimate (hot-cell births;
EXPLORATORY stratification of H5). Region and overall inverse-variance pooled estimates with I^2.
Illustrative burden: extra neonatal deaths per year for a sustained +1 degC in hot cells
= hot-tertile estimate for both sexes (pp per degC) x WPP 2023 births x share of births in hot cells.
"""
import os, json, numpy as np, pandas as pd, pycountry
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); RES = os.path.join(HERE, 'results')

B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=['cc', 'primary', 'clim_tertile', 'age_m', 'b5', 'b7'])
B = B[B.primary]
B['nnd'] = ((B.b5 == 0) & (B.b7 == 0)).astype(float)
g = B.groupby('cc').agg(births=('nnd', 'size'), hot_share=('clim_tertile', lambda s: (s == 2).mean()))
g['nnd_hot_per1000'] = B[B.clim_tertile == 2].groupby('cc').nnd.mean() * 1000
S = pd.read_csv(os.path.join(DATA, 'dhs_api_surveys.csv'))
meta = S.drop_duplicates('DHS_CountryCode').set_index('DHS_CountryCode')[['CountryName', 'RegionName', 'SubregionName']]
g = g.join(meta, how='left')
FIX = {'Congo Democratic Republic': 'COD', 'Congo': 'COG', 'Cote d\'Ivoire': 'CIV', 'Bolivia': 'BOL', 'Tanzania': 'TZA',
       'Kyrgyz Republic': 'KGZ', 'Moldova': 'MDA', 'Eswatini': 'SWZ', 'Turkey': 'TUR', 'Timor-Leste': 'TLS',
       'Gambia': 'GMB', 'Lao People\'s Democratic Republic': 'LAO', 'Burkina Faso': 'BFA'}
def iso3(n):
    if n in FIX: return FIX[n]
    try: return pycountry.countries.lookup(n).alpha_3
    except LookupError: return pycountry.countries.search_fuzzy(n)[0].alpha_3
g['iso3'] = g.CountryName.map(iso3)
# sub-Saharan Africa split by DHS subregion; other DHS regions kept
g['region'] = np.where(g.RegionName == 'Sub-Saharan Africa', g.SubregionName.fillna('Sub-Saharan Africa'), g.RegionName)
M = pd.read_csv(os.path.join(RES, 'fig_country.csv')); M = M[M.outcome == 'male'].set_index('cc')[['b', 'se']].add_prefix('male_')
N = pd.read_csv(os.path.join(RES, 'fig_country_nnd_hot.csv')).set_index('cc')[['b', 'se', 'n']].rename(columns={'b': 'nnd_b', 'se': 'nnd_se', 'n': 'nnd_hot_births'})
T = g.join(M).join(N)
W = pd.read_csv(os.path.join(DATA, 'wpp_demo.csv.gz'), usecols=['ISO3_code', 'Time', 'Births'], low_memory=False)
W = W[(W.Time == 2023) & W.ISO3_code.notna()].groupby('ISO3_code').Births.first() * 1000   # thousands -> births
T['wpp_births_2023'] = T.iso3.map(W)
# both-sexes hot-tertile estimate (analysis_h5_pooled.py); the H5 interaction model's GEST term is the effect in girls
h5 = pd.read_csv(os.path.join(RES, 'h5_pooled.csv'))
h5 = h5[(h5.model == 'nnd (hot), both sexes') & (h5.term == 'T2M_anom_GEST')].iloc[0]
for k, v in [('extra_nnd_per_year', h5.b_pp), ('extra_nnd_lo', h5.lo_pp), ('extra_nnd_hi', h5.hi_pp)]:
    T[k] = v / 100 * T.wpp_births_2023 * T.hot_share
T.to_csv(os.path.join(RES, 'country_table.csv'))


def pool(b, se):
    w = 1 / se ** 2; est = (w * b).sum() / w.sum(); q = (w * (b - est) ** 2).sum(); k = len(b)
    i2 = max(0, (q - (k - 1)) / q) * 100 if q > 0 else 0
    return est, np.sqrt(1 / w.sum()), i2, k
rows = []
for col in ['male', 'nnd']:
    D = T.dropna(subset=[f'{col}_b'])
    for reg, d in list(D.groupby('region')) + [('Overall', D)]:
        e, s, i2, k = pool(d[f'{col}_b'], d[f'{col}_se']); rows.append(dict(outcome=col, region=reg, b=e, se=s, lo=e - 1.96 * s, hi=e + 1.96 * s, I2=i2, k=k))
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, 'country_pooled.csv'), index=False)
pd.set_option('display.width', 220)
print(T[['CountryName', 'region', 'births', 'hot_share', 'male_b', 'nnd_b', 'wpp_births_2023', 'extra_nnd_per_year']].sort_values('region').round(3).to_string())
print(R.round(3).to_string(index=False))
print('total extra neonatal deaths/yr per +1C (hot cells, 56 countries):', round(T.extra_nnd_per_year.sum()), '(', round(T.extra_nnd_lo.sum()), '-', round(T.extra_nnd_hi.sum()), ')')
