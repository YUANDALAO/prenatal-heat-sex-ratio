"""Lancet-style country-level main figures.
FigL1: maps - country estimate for neonatal death (hot-cell births) and illustrative extra neonatal deaths per +1 degC.
FigL2: country forest plot grouped by region, two outcomes side by side, with region and overall pooled diamonds.
FigL3: illustrative burden by country (extra neonatal deaths per year per sustained +1 degC in hot cells).
Diverging colour: blue -> neutral #f0efec -> red centred at 0. Burden: single red hue.
"""
import os, json
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import figs_heat as fh

RES, DATA, FIG, MM = fh.RES, fh.DATA, fh.FIG, fh.MM
INK, INK2, GRAY, RED, BLUE = fh.INK, fh.INK2, fh.GRAY, '#e34948', '#2a78d6'
DIV = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
T = pd.read_csv(os.path.join(RES, 'country_table.csv')).rename(columns={'Unnamed: 0': 'cc'})
R = pd.read_csv(os.path.join(RES, 'country_pooled.csv'))
GEO = json.load(open(os.path.join(DATA, 'ne_110m_countries.geojson'), encoding='utf-8'))
REG_ORDER = ['Western Africa', 'Middle Africa', 'Eastern Africa', 'Southern Africa', 'North Africa/West Asia/Europe',
             'Central Asia', 'South & Southeast Asia', 'Latin America & Caribbean']
SHORT = {'Congo Democratic Republic': 'DR Congo', 'Central African Republic': 'Central African Rep.', 'Dominican Republic': 'Dominican Rep.',
         'Kyrgyz Republic': 'Kyrgyzstan', "Cote d'Ivoire": "Côte d'Ivoire"}


def polys(feat):
    g = feat['geometry']; parts = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    return [np.array(p[0]) for p in parts]


def draw_map(ax, values, cmap, norm, missing='#cfcdc8', nodata='#f7f6f3'):
    in_sample = set(T.iso3)
    for f in GEO['features']:
        iso = f['properties'].get('ADM0_A3') or f['properties'].get('ISO_A3')
        v = values.get(iso, np.nan)
        col = cmap(norm(v)) if np.isfinite(v) else (missing if iso in in_sample else nodata)
        for p in polys(f): ax.add_patch(Polygon(p, closed=True, facecolor=col, edgecolor='white', linewidth=0.3))
    ax.set_xlim(-95, 160); ax.set_ylim(-36, 42); ax.set_aspect('equal'); ax.axis('off')


def centroid(iso):
    for f in GEO['features']:
        if (f['properties'].get('ADM0_A3') or f['properties'].get('ISO_A3')) == iso:
            p = max(polys(f), key=len); return p[:, 0].mean(), p[:, 1].mean()
    return np.nan, np.nan


def figL1():
    fig, axs = plt.subplots(2, 1, figsize=(180 * MM, 150 * MM), gridspec_kw={'hspace': 0.12})
    v = dict(zip(T.iso3, T.nnd_b)); norm = TwoSlopeNorm(0, -1.5, 1.5)
    draw_map(axs[0], v, DIV, norm)
    sm = mpl.cm.ScalarMappable(cmap=DIV, norm=norm)
    cb = fig.colorbar(sm, ax=axs[0], orientation='horizontal', fraction=0.04, pad=0.02, shrink=0.45, aspect=30)
    cb.set_label('Change in neonatal death (pp) per 1 °C gestational anomaly, births in hot cells', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    axs[0].text(0.0, 1.0, 'A   Country estimates, neonatal death (27 countries with ≥15 000 births in hot cells)', transform=axs[0].transAxes, fontsize=7.5, fontweight='bold', va='bottom')
    axs[0].text(0.0, 0.0, 'Dark grey: in sample, too few births in hot cells\nLight: not in sample', transform=axs[0].transAxes, fontsize=6, color=INK2, va='bottom')
    ax = axs[1]; draw_map(ax, {}, DIV, norm, missing='#e4e2dd')
    d = T[T.extra_nnd_per_year > 0].copy(); d[['x', 'y']] = d.iso3.apply(lambda i: pd.Series(centroid(i)))
    ax.scatter(d.x, d.y, s=d.extra_nnd_per_year / 22, color=RED, alpha=0.75, edgecolor='white', linewidth=0.5, zorder=3)
    OFF = {'Nigeria': (-8, -16), 'Indonesia': (0, -14), 'Niger': (6, 14), 'Philippines': (8, 9), 'Mali': (-16, 14),
           'Chad': (20, 6), 'Burkina Faso': (-30, 3), 'Bangladesh': (-6, 13)}
    for _, r in d.nlargest(8, 'extra_nnd_per_year').iterrows():
        nm = SHORT.get(r.CountryName, r.CountryName); ox, oy = OFF.get(nm, (0, 10))
        ax.annotate(f"{nm} {r.extra_nnd_per_year:,.0f}", (r.x, r.y), xytext=(r.x + ox, r.y + oy), fontsize=5.8, color=INK, ha='center', va='center', zorder=4,
                    arrowprops=dict(arrowstyle='-', lw=0.4, color=INK2, shrinkA=0, shrinkB=1))
    for s_ in [500, 2000, 6000]: ax.scatter([], [], s=s_ / 22, color=RED, alpha=0.75, edgecolor='white', label=f'{s_:,}')
    ax.legend(title='Extra neonatal deaths per year', loc='lower left', bbox_to_anchor=(0.0, 0.02), ncol=3, fontsize=6, title_fontsize=6.3,
              frameon=False, labelspacing=1.2, handletextpad=0.8, columnspacing=1.6, borderpad=0.8)
    tot, lo, hi = T.extra_nnd_per_year.sum(), T.extra_nnd_lo.sum(), T.extra_nnd_hi.sum()
    ax.text(0.0, 1.0, f'B   Illustrative burden of a sustained +1 °C in hot cells: {tot:,.0f} (95% CI {lo:,.0f}–{hi:,.0f}) extra neonatal deaths per year', transform=ax.transAxes, fontsize=7.5, fontweight='bold', va='bottom')
    fh.save(fig, 'FigL1_maps')


def diamond(ax, y, b, lo, hi, color, h=0.32):
    ax.add_patch(Polygon([[lo, y], [b, y + h], [hi, y], [b, y - h]], closed=True, facecolor=color, edgecolor=color, zorder=3))


def figL2():
    rows = []
    for reg in REG_ORDER:
        d = T[T.region == reg].sort_values('births', ascending=False)
        if d.empty: continue
        rows.append(('region', reg, None))
        for _, r in d.iterrows(): rows.append(('country', SHORT.get(r.CountryName, r.CountryName), r))
        rows.append(('pool', reg, None))
    rows.append(('overall', 'All countries', None))
    n = len(rows)
    fig = plt.figure(figsize=(180 * MM, (n * 3.15 + 22) * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.15, 1.15], wspace=0.05)
    axT, axM, axN = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    for ax in (axT, axM, axN): ax.set_ylim(n - 0.5, -1.6); ax.grid(False)
    axT.axis('off')
    hdr = -1.2
    axT.text(0.0, hdr, 'Country', fontweight='bold', fontsize=6.5); axT.text(0.56, hdr, 'Births', fontweight='bold', fontsize=6.5, ha='right')
    axT.text(0.76, hdr, '% hot', fontweight='bold', fontsize=6.5, ha='right'); axT.text(1.0, hdr, 'NND‰ hot', fontweight='bold', fontsize=6.5, ha='right')
    for ax, col, title, xl in [(axM, 'male', 'Male share of births', (-4.5, 4.5)), (axN, 'nnd', 'Neonatal death (births in hot cells)', (-2.2, 2.2))]:
        ax.axvline(0, color=GRAY, lw=0.7); ax.set_xlim(*xl); ax.spines['left'].set_visible(False); ax.set_yticks([])
        ax.set_title(title, fontsize=7, fontweight='bold', loc='center', pad=20)
        ax.set_xlabel('pp per 1 °C gestational anomaly', fontsize=6.5); ax.tick_params(labelsize=6)
        for i, (kind, label, r) in enumerate(rows):
            if kind == 'country' and np.isfinite(r[f'{col}_b']):
                b, se = r[f'{col}_b'], r[f'{col}_se']; lo, hi = b - 1.96 * se, b + 1.96 * se
                ax.plot([max(lo, xl[0]), min(hi, xl[1])], [i, i], color=INK2, lw=0.9)
                ax.plot([b], [i], marker='s', ms=2.2 + 3 * np.sqrt(r.births / T.births.max()), color=INK, zorder=3)
                if lo < xl[0]: ax.annotate('', (xl[0], i), (xl[0] + 0.3, i), arrowprops=dict(arrowstyle='->', lw=0.6, color=INK2))
                if hi > xl[1]: ax.annotate('', (xl[1], i), (xl[1] - 0.3, i), arrowprops=dict(arrowstyle='->', lw=0.6, color=INK2))
            elif kind in ('pool', 'overall'):
                reg = label if kind == 'pool' else 'Overall'
                p = R[(R.outcome == col) & (R.region == reg)]
                if len(p): p = p.iloc[0]; diamond(ax, i, p.b, p.lo, p.hi, RED if kind == 'overall' else '#9a9994')
    for i, (kind, label, r) in enumerate(rows):
        if kind == 'region': axT.text(0.0, i, label, fontsize=6.5, fontweight='bold', va='center')
        elif kind == 'country':
            axT.text(0.03, i, label, fontsize=6.2, va='center'); axT.text(0.56, i, f'{int(r.births):,}', fontsize=6.2, va='center', ha='right')
            axT.text(0.76, i, f'{r.hot_share * 100:.0f}', fontsize=6.2, va='center', ha='right')
            if np.isfinite(r.nnd_hot_per1000): axT.text(1.0, i, f'{r.nnd_hot_per1000:.0f}', fontsize=6.2, va='center', ha='right')
        elif kind == 'pool': axT.text(0.03, i, 'Regional pooled estimate', fontsize=6.2, va='center', style='italic', color=INK2)
        else: axT.text(0.0, i, 'All countries, pooled', fontsize=6.6, fontweight='bold', va='center')
    for ax, col in [(axM, 'male'), (axN, 'nnd')]:
        p = R[(R.outcome == col) & (R.region == 'Overall')].iloc[0]
        ax.text(0.5, 1.0, f'pooled {p.b:+.2f} ({p.lo:+.2f} to {p.hi:+.2f})' + chr(10) + f'I² {p.I2:.0f}%, {int(p.k)} countries',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=6, color=INK2)
    for ax in (axM, axN): ax.spines['bottom'].set_position(('data', n - 0.5))
    fh.save(fig, 'FigL2_country_forest')


def figL3():
    d = T[T.extra_nnd_per_year > 1].sort_values('extra_nnd_per_year')
    fig, ax = plt.subplots(figsize=(120 * MM, (len(d) * 3.2 + 20) * MM)); ax.grid(True, axis='x'); ax.grid(False, axis='y')
    y = np.arange(len(d))
    ax.hlines(y, d.extra_nnd_lo, d.extra_nnd_hi, color='#f0b3b2', lw=2.2)
    ax.plot(d.extra_nnd_per_year, y, 'o', color=RED, ms=3.6, mec='white', mew=0.5)
    ax.set_yticks(y); ax.set_yticklabels([SHORT.get(n, n) for n in d.CountryName], fontsize=6.2)
    for yi, (_, r) in zip(y, d.iterrows()): ax.text(r.extra_nnd_hi + 120, yi, f'{r.extra_nnd_per_year:,.0f}', va='center', fontsize=5.8, color=INK2)
    ax.set_xlabel('Extra neonatal deaths per year for a sustained +1 °C in hot cells (95% CI)')
    tot, lo, hi = T.extra_nnd_per_year.sum(), T.extra_nnd_lo.sum(), T.extra_nnd_hi.sum()
    ax.set_title(f'Total {tot:,.0f} (95% CI {lo:,.0f}–{hi:,.0f}) per year, {len(d)} countries', loc='left', fontsize=7)
    fh.save(fig, 'FigL3_burden')


if __name__ == '__main__':
    figL1(); figL2(); figL3()
