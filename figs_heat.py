"""Figures 1-4 for 'Prenatal heat scars rather than culls'. Output: figures/Fig*.pdf + .png (300 dpi).
Colour-blind-safe palette (adjacent pairs separable under common colour-vision deficiencies; yellow has low contrast, so every series is directly labelled):
cool #2a78d6, temperate #eda100, hot #e34948.
"""
import os, json
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'results'); DATA = os.path.join(HERE, 'data')
FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
COL = {0: '#2a78d6', 1: '#eda100', 2: '#e34948'}; NAME = {0: 'Cool', 1: 'Temperate', 2: 'Hot'}
INK, INK2, GRID, GRAY = '#0b0b0b', '#52514e', '#e6e5e1', '#9a9994'
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.fontsize': 6.5, 'pdf.fonttype': 42, 'lines.linewidth': 1.5})
MM = 1 / 25.4


def tag(ax, s): ax.text(-0.12, 1.04, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')


def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig); print('saved', name)


def ci(ax, y, b, lo, hi, color, size=4.5, marker='o'):
    ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle='round', zorder=2)
    ax.plot([b], [y], marker=marker, ms=size, color=color, mec='white', mew=0.8, zorder=3)


P = pd.read_csv(os.path.join(RES, 'primary_results.csv'))
cuts = json.load(open(os.path.join(DATA, 'climate_tertiles.json')))['tertile_cuts_degC']


def fig1():
    ds = xr.open_zarr(os.path.join(DATA, 'power_monthly_anom.zarr')); LAT, LON = ds.lat.values, ds.lon.values
    clim = ds.T2M.sel(time=slice('1991-01-01', '2020-12-31')).mean('time').values.ravel()
    C = pd.read_csv(os.path.join(RES, 'fig_map_cells.csv'))
    C['lat'] = LAT[C.cell // len(LON)]; C['lon'] = LON[C.cell % len(LON)]; C['clim'] = clim[C.cell]
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 70 * MM), gridspec_kw={'width_ratios': [2.3, 1], 'wspace': 0.18})
    ax = axs[0]; ax.grid(False); ax.set_aspect('equal')
    for t in range(3):
        d = C[C.clim_tertile == t]
        ax.scatter(d.lon, d.lat, s=1.5 + 10 * np.sqrt(d.births / C.births.max()), color=COL[t], linewidth=0, alpha=0.85,
                   label=f'{NAME[t]} ({d.births.sum() / 1e6:.2f} M births)')
    ax.set_xlim(-95, 130); ax.set_ylim(-36, 45); ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, markerscale=3, handletextpad=0.3, columnspacing=1.2)
    ax.set_title('Grid cells contributing births (primary sample, 56 countries)', loc='left'); tag(ax, 'a')
    ax = axs[1]; w = C.groupby(pd.cut(C.clim, np.arange(5, 32, 1))).births.sum()
    x = np.arange(5.5, 31, 1)
    ax.bar(x, w.values / 1e3, width=0.85, color=[COL[int(np.digitize(v, cuts))] for v in x], edgecolor='white', linewidth=0.6)
    for c in cuts: ax.axvline(c, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ax.text(cuts[0] - 0.3, ax.get_ylim()[1] * 0.97, f'{cuts[0]:.1f} °C', ha='right', va='top', fontsize=6.3, color=INK2)
    ax.text(cuts[1] + 0.3, ax.get_ylim()[1] * 0.97, f'{cuts[1]:.1f} °C', ha='left', va='top', fontsize=6.3, color=INK2)
    ax.set_xlabel('Cell mean temperature, 1991–2020 (°C)'); ax.set_ylabel('Births (thousands)')
    ax.set_title('Baseline climate tertiles', loc='left'); tag(ax, 'b'); save(fig, 'Fig1_sample')


def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 68 * MM), gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.62})
    ax = axs[0]; ax.grid(True, axis='x'); ax.grid(False, axis='y')
    h2 = P[P.model.str.startswith('H2: slopes')].set_index('term'); h1 = P[P.model.str.startswith('H1 primary')].set_index('term')
    ax.axvspan(-1.2, 0, color='#fbe9e8', lw=0, zorder=0)
    ax.text(-1.15, 3.55, 'Culling prediction\nfor hot climates', fontsize=6.2, color=INK2, va='top')
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.axvline(0.25, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.text(0.27, 3.55, 'Sami, cold climate:\n≈ +0.25 pp per °C', fontsize=6.2, color=INK2, va='top')
    rows = [(f'{NAME[t]} (H2)', h2.loc[f'G{t}'], COL[t], 'o') for t in range(3)] + [('Hot, primary test (H1)', h1.loc['T2M_anom_GEST'], COL[2], 'D')]
    for i, (lab, r, c, mk) in enumerate(rows[::-1]):
        ci(ax, i, r.b_pp, r.lo_pp, r.hi_pp, c, marker=mk)
        ax.text(1.03, i, f'{r.b_pp:+.2f} ({r.lo_pp:+.2f}, {r.hi_pp:+.2f})', transform=ax.get_yaxis_transform(), fontsize=6.2, va='center', ha='left', color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.6, 3.7)
    ax.set_xlabel('Change in male share of births (pp) per 1 °C gestational anomaly')
    ax.set_title('No shift in the sex ratio at birth', loc='left'); tag(ax, 'a')
    ax = axs[1]
    centers = {'bin_lt_m1': -1.25, 'bin_m1_m05': -0.75, 'ref': 0, 'bin_p05_p1': 0.75, 'bin_gt_p1': 1.25}
    ax.axhline(0, color=GRAY, lw=0.8)
    for t in range(3):
        d = P[P.model == f'H3 bins, tertile {t}'].set_index('term')
        xs, bs, los, his = [0], [0], [0], [0]
        for k in ['bin_lt_m1', 'bin_m1_m05', 'bin_p05_p1', 'bin_gt_p1']:
            xs.append(centers[k]); bs.append(d.loc[k, 'b_pp']); los.append(d.loc[k, 'lo_pp']); his.append(d.loc[k, 'hi_pp'])
        o = np.argsort(xs); xs, bs, los, his = (np.array(v)[o] for v in (xs, bs, los, his)); dx = (t - 1) * 0.07
        ax.fill_between(xs + dx, los, his, color=COL[t], alpha=0.10, lw=0)
        ax.plot(xs + dx, bs, color=COL[t], marker='o', ms=3.5, mec='white', mew=0.6)
        ax.text(1.42, bs[-1], NAME[t], color=INK, fontsize=6.3, va='center')
    ax.set_xticks([-1.25, -0.75, 0, 0.75, 1.25]); ax.set_xticklabels(['< −1', '−1 to −0.5', '−0.5 to 0.5\n(reference)', '0.5 to 1', '> 1'])
    ax.set_xlim(-1.5, 1.75); ax.set_xlabel('Gestational temperature anomaly (°C)'); ax.set_ylabel('Male share vs reference (pp)')
    ax.set_title('No dose–response in any climate (H3)', loc='left'); tag(ax, 'b'); save(fig, 'Fig2_sex_ratio')


def fig3():
    F = pd.read_csv(os.path.join(RES, 'fig_data_h5.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 66 * MM), gridspec_kw={'wspace': 0.42})
    for ax, out, title, lab in [(axs[0], 'nnd', 'Neonatal death', 'a'), (axs[1], 'imr', 'Infant death', 'b')]:
        ax.grid(True, axis='x'); ax.grid(False, axis='y'); ax.axvline(0, color=GRAY, lw=0.8)
        d = F[F.outcome == out].set_index('group')
        rows = [('Cool (exploratory)', 'tertile 0', COL[0]), ('Temperate (exploratory)', 'tertile 1', COL[1]),
                ('Hot (pre-specified)', 'tertile 2', COL[2]), ('Hot, girls', 'hot, female', COL[2]), ('Hot, boys', 'hot, male', COL[2])]
        for i, (name, key, c) in enumerate(rows[::-1]):
            r = d.loc[key]; mk = 'o' if 'tertile' in key else ('s' if 'female' in key else '^')
            ci(ax, i, r.b, r.lo, r.hi, c, marker=mk)
        ax.axhline(1.5, color=GRID, lw=0.8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlim(-0.45, 0.6)
        ax.set_xlabel('pp per 1 °C gestational anomaly')
        ax.set_title(title, loc='left'); tag(ax, lab)
    save(fig, 'Fig3_mortality')


def fig4():
    R = pd.read_csv(os.path.join(RES, 'robustness.csv'))
    labels = {'0 primary specification': 'Primary specification', '1 Conley 100 km SE': 'Conley 100 km SE',
              '2 sampling weights (v005)': 'Sampling weights', '3 non-movers only': 'Non-movers only',
              '4 births <=60 months before interview': 'Births ≤ 5 y before interview',
              '5 survey x birth-year + country x month FE': 'Survey × birth-year FE',
              '7 pre-conception window added': '+ pre-conception window'}
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 66 * MM), sharey=True, gridspec_kw={'wspace': 0.12})
    for ax, out, title, lab in [(axs[0], 'H1 male', 'Male share (H1)', 'a'), (axs[1], 'H5 neonatal death', 'Neonatal death (H5)', 'b'),
                                (axs[2], 'H5 infant death', 'Infant death (H5)', 'c')]:
        ax.grid(True, axis='x'); ax.grid(False, axis='y'); ax.axvline(0, color=GRAY, lw=0.8)
        d = R[R.outcome == out]; keys = list(labels)
        for i, k in enumerate(keys[::-1]):
            r = d[d.check == k].iloc[0]; ci(ax, i + 1, r.b_pp, r.lo_pp, r.hi_pp, COL[2] if out != 'H1 male' else INK2)
        L = d[d.check.str.startswith('6')]
        ax.plot([L.b_pp.min(), L.b_pp.max()], [0, 0], color=COL[2] if out != 'H1 male' else INK2, lw=4, alpha=0.45, solid_capstyle='round')
        ax.set_title(title, loc='left'); tag(ax, lab); ax.set_xlabel('pp per 1 °C (hot tertile)')
    axs[0].set_yticks(range(len(labels) + 1)); axs[0].set_yticklabels(['Leave-one-country-out\n(range of 45 estimates)'] + [labels[k] for k in list(labels)[::-1]])
    save(fig, 'Fig4_robustness')


if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4()
