"""Dense multi-panel main figures (replace the coefficient-only figures; robustness forest moves to ESM).
Fig1 data & design (5 panels); Fig2 sex ratio (4 panels); Fig3 neonatal mortality (4 panels).
Categorical palette (validated): cool #2a78d6, temperate #eda100, hot #e34948; all series directly labelled.
Diverging heat maps: blue -> neutral gray #f0efec -> red, centred at 0.
"""
import os, json
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, xarray as xr
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import figs_heat as fh                                  # shared style, colours, helpers

RES, DATA, FIG = fh.RES, fh.DATA, fh.FIG; COL, NAME = fh.COL, fh.NAME; INK, INK2, GRAY = fh.INK, fh.INK2, fh.GRAY
MM = fh.MM; tag = fh.tag; save = fh.save
DIV = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#e34948'])
cuts = json.load(open(os.path.join(DATA, 'climate_tertiles.json')))['tertile_cuts_degC']


def gest_band(ax):
    ax.axvspan(-9.5, -0.5, color='#f6f5f2', lw=0, zorder=0)
    ax.text(-5, 1.0, 'gestation', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.2, color=INK2)


def binscatter(ax, d, ycol, secol, color, label, scale=100, dx=0.0, marker='o', ly=0.0):
    x = d.x.to_numpy() + dx; y = d[ycol].to_numpy() * scale; e = d[secol].to_numpy() * scale * 1.96
    ax.errorbar(x, y, yerr=e, fmt='none', ecolor=color, elinewidth=0.8, alpha=0.6, capsize=0)
    ax.plot(x, y, marker=marker, ls='none', ms=3.8, color=color, mec='white', mew=0.5)
    b = np.polyfit(d.x, d[ycol] * scale, 1, w=d.n); xx = np.array([d.x.min(), d.x.max()])
    ax.plot(xx + dx, np.polyval(b, xx), color=color, lw=1.2)
    ax.text(xx[1] + 0.03, np.polyval(b, xx[1]) + ly, label, fontsize=6.3, color=INK, va='center')


def heat(ax, H, col, title, vlim, unit):
    piv = H.pivot(index='ab', columns='cb', values=col) * 100
    rows = ['< −0.75', '−0.75 to −0.25', '±0.25', '0.25 to 0.75', '> 0.75']; colsx = ['<16', '16–19', '19–21', '21–23', '23–24.5', '24.5–26', '26–27', '>27']
    piv = piv.reindex(index=rows[::-1], columns=colsx)
    n = H.pivot(index='ab', columns='cb', values='n').reindex(index=rows[::-1], columns=colsx)
    im = ax.imshow(piv.values, cmap=DIV, norm=TwoSlopeNorm(0, -vlim, vlim), aspect='auto')
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                small = n.values[i, j] < 50000
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=5.4, color=INK2 if small else INK, style='italic' if small else 'normal')
                if small: ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='white', alpha=0.55, lw=0))
    ax.set_xticks(range(len(colsx))); ax.set_xticklabels(colsx, rotation=35, ha='right')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows[::-1]); ax.grid(False)
    ax.axvline(3.5, color=INK2, lw=0.6, ls=(0, (3, 2))); ax.axvline(5.5, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel('Cell mean temperature, 1991–2020 (°C)'); ax.set_ylabel('Gestational anomaly (°C)')
    ax.set_title(title, loc='left')
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label(unit, fontsize=6.3); cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)


def lagprof(ax, L, outcome, tertiles, ylabel):
    """3-month exposure blocks (EXPLORATORY); points at block midpoints, horizontal bar = block span."""
    gest_band(ax); ax.axhline(0, color=GRAY, lw=0.8)
    for k, t in enumerate(tertiles):
        d = L[(L.outcome == outcome) & (L.tertile == t)]; dx = (k - 0.5) * 0.6
        for _, r in d.iterrows():
            ax.plot([r.mid + dx] * 2, [r.lo_pp, r.hi_pp], color=COL[t], lw=1.1, alpha=0.8)
        ax.plot(d.mid + dx, d.b_pp, color=COL[t], lw=0.9, ls=(0, (2, 1.5)), marker='o', ms=4, mec='white', mew=0.6)
        ax.text(11.8, d.b_pp.iloc[-1], NAME[t], fontsize=6.3, va='center', color=INK)
    ax.axvline(-0.5, color=INK2, lw=0.6); ax.text(-0.3, 0.02, 'birth', transform=ax.get_xaxis_transform(), fontsize=6, color=INK2)
    ax.text(-14, 1.0, 'pre-conception', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=6.2, color=INK2)
    ax.set_xticks([-17, -14, -11, -8, -5, -2, 1, 4, 7, 10])
    ax.set_xticklabels(['−18', '−15', '−12', 'T1', 'T2', 'T3', '0–2', '3–5', '6–8', '9–11'], fontsize=6)
    ax.set_xlim(-19, 14.5); ax.set_xlabel('3-month exposure block (months relative to birth)'); ax.set_ylabel(ylabel)


def funnel(ax, C, outcome, pooled, title):
    if outcome == 'nnd': C = pd.read_csv(os.path.join(RES, 'fig_country_nnd_hot.csv')); outcome = 'nnd_hot'
    d = C[C.outcome == outcome].copy(); d['prec'] = 1 / d.se
    pooled = float((d.b / d.se ** 2).sum() / (1 / d.se ** 2).sum())    # inverse-variance pooled country estimate
    ax.axvline(0, color=GRAY, lw=0.8); ax.axvline(pooled, color=COL[2], lw=1, ls=(0, (3, 2)))
    pr = np.linspace(d.prec.min() * 0.8, d.prec.max() * 1.05, 50)
    ax.fill_betweenx(pr, pooled - 1.96 / pr, pooled + 1.96 / pr, color='#f6f5f2', lw=0, zorder=0)
    ax.scatter(d.b, d.prec, s=10 + 30 * np.sqrt(d.n / d.n.max()), color=INK2, alpha=0.75, linewidth=0.4, edgecolor='white', zorder=3)
    for _, r in d.nlargest(4, 'prec').iterrows(): ax.text(r.b, r.prec, ' ' + r.cc, fontsize=5.8, color=INK2, va='center')
    ax.set_xlabel('Country estimate (pp per °C)'); ax.set_ylabel('Precision (1/SE)'); ax.grid(True, axis='both')
    ax.text(0.98, 0.04, f'pooled {pooled:+.2f}; {len(d)} countries', transform=ax.transAxes, ha='right', fontsize=6.2, color=INK2)
    ax.set_title(title, loc='left')


def fig1():
    ds = xr.open_zarr(os.path.join(DATA, 'power_monthly_anom.zarr')); LAT, LON = ds.lat.values, ds.lon.values
    clim = ds.T2M.sel(time=slice('1991-01-01', '2020-12-31')).mean('time').values.ravel()
    C = pd.read_csv(os.path.join(RES, 'fig_map_cells.csv')); C['lat'] = LAT[C.cell // len(LON)]; C['lon'] = LON[C.cell % len(LON)]
    C['clim'] = clim[C.cell]
    A = pd.read_csv(os.path.join(RES, 'fig_anom_sample.csv')); V = pd.read_csv(os.path.join(RES, 'fig_variance.csv'))
    fig = plt.figure(figsize=(180 * MM, 125 * MM)); gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.55, wspace=0.55)
    ax = fig.add_subplot(gs[0, :2]); ax.grid(False); ax.set_aspect('equal')
    for t in range(3):
        d = C[C.clim_tertile == t]
        ax.scatter(d.lon, d.lat, s=1.2 + 9 * np.sqrt(d.births / C.births.max()), color=COL[t], linewidth=0, alpha=0.85,
                   label=f'{NAME[t]}: {d.births.sum() / 1e6:.2f} M births')
    ax.set_xlim(-95, 130); ax.set_ylim(-36, 45); ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, markerscale=3, handletextpad=0.3, columnspacing=1.2)
    ax.set_title('4.38 M births in 102,930 clusters, 56 countries (dot = 0.5° grid cell)', loc='left'); tag(ax, 'a')
    ax = fig.add_subplot(gs[0, 2]); x = np.arange(5.5, 31, 1)
    w = C.groupby(pd.cut(C.clim, np.arange(5, 32, 1)), observed=False).births.sum()
    ax.bar(x, w.values / 1e3, width=0.85, color=[COL[int(np.digitize(v, cuts))] for v in x], edgecolor='white', linewidth=0.5)
    for c in cuts: ax.axvline(c, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel('Cell mean temperature (°C)'); ax.set_ylabel('Births (thousands)'); ax.set_title('Baseline climate', loc='left'); tag(ax, 'b')
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-2.5, 2.5, 61)
    for t in range(3):
        h, e = np.histogram(A[A.clim_tertile == t].T2M_anom_GEST, bins=bins, density=True)
        ax.plot((e[:-1] + e[1:]) / 2, h, color=COL[t], lw=1.4, label=NAME[t])
    ax.legend(loc='upper right', handlelength=1.2); ax.axvline(0, color=GRAY, lw=0.7); ax.set_xlabel('Gestational anomaly, GEST (°C)'); ax.set_ylabel('Density')
    ax.set_title('Exposure distribution', loc='left'); tag(ax, 'c')
    ax = fig.add_subplot(gs[1, 1]); ax.grid(True, axis='x'); ax.grid(False, axis='y')
    for t in range(3):
        r = V[V.tertile == t].iloc[0]
        ax.barh(t, r.within_share * 100, height=0.5, color=COL[t], edgecolor='white')
        ax.text(r.within_share * 100 + 2, t, f'{r.within_share * 100:.0f}%\nSD {r.sd:.2f} °C', va='center', fontsize=6.0)
    ax.set_yticks(range(3)); ax.set_yticklabels([NAME[t] for t in range(3)]); ax.set_xlim(0, 115)
    ax.set_xlabel('Variance of GEST within clusters (%)'); ax.set_title('Identifying variation', loc='left'); tag(ax, 'd')
    ax = fig.add_subplot(gs[1, 2])
    top = C.sort_values('births', ascending=False).groupby('clim_tertile').head(1)
    for _, r in top.iterrows():
        s = ds.T2M_anom.isel(lat=int(r.cell // len(LON)), lon=int(r.cell % len(LON))).sel(time=slice('2005', '2014')).values
        s9 = pd.Series(s).rolling(9).mean()
        ax.plot(np.arange(len(s)) / 12 + 2005, s9, color=COL[int(r.clim_tertile)], lw=1.1)
    ax.axhline(0, color=GRAY, lw=0.7); ax.set_xticks([2006, 2008, 2010, 2012, 2014]); ax.set_xlabel('Year of birth'); ax.set_ylabel('GEST (°C)')
    ax.set_title('GEST in the largest cell per tertile', loc='left'); tag(ax, 'e')
    save(fig, 'Fig1_data_design')


def fig23(outcome, fname, title_a, ylab, vlim, unit, pooled, ftitle):
    Bs = pd.read_csv(os.path.join(RES, 'fig2_binscatter.csv')); H = pd.read_csv(os.path.join(RES, 'fig_heatmap.csv'))
    L = pd.read_csv(os.path.join(RES, 'fig_lagblocks.csv')); Cn = pd.read_csv(os.path.join(RES, 'fig_country.csv'))
    fig, axs = plt.subplots(2, 2, figsize=(180 * MM, 130 * MM), gridspec_kw={'hspace': 0.62, 'wspace': 0.38})
    ax = axs[0, 0]; ax.axhline(0, color=GRAY, lw=0.8)
    ycol, secol = ('m', 'm_se') if outcome == 'male' else ('d', 'd_se')
    if outcome == 'male':
        for t in range(3): binscatter(ax, Bs[(Bs.tertile == t) & (Bs.sex == 'all')], ycol, secol, COL[t], NAME[t], dx=(t - 1) * 0.004, ly=(t - 1) * 0.09)
    else:
        binscatter(ax, Bs[(Bs.tertile == 2) & (Bs.sex == 'girls')], ycol, secol, COL[2], 'Hot, girls', marker='s')
        binscatter(ax, Bs[(Bs.tertile == 2) & (Bs.sex == 'boys')], ycol, secol, INK2, 'Hot, boys', marker='^', dx=0.006)
        binscatter(ax, Bs[(Bs.tertile == 0) & (Bs.sex == 'all')], ycol, secol, COL[0], 'Cool, all', dx=-0.006)
    ax.set_xlabel('GEST, residual (°C)'); ax.set_ylabel(ylab); ax.set_title(title_a, loc='left'); ax.set_xlim(None, ax.get_xlim()[1] + 0.25); tag(ax, 'a')
    heat(axs[0, 1], H, outcome, 'Baseline climate × anomaly', vlim, unit); tag(axs[0, 1], 'b')
    lagprof(axs[1, 0], L, outcome, [2, 0], 'pp per 1 °C in the block'); axs[1, 0].set_title('Timing of exposure (exploratory)', loc='left'); tag(axs[1, 0], 'c')
    funnel(axs[1, 1], Cn, outcome, pooled, ftitle); tag(axs[1, 1], 'd')
    save(fig, fname)


if __name__ == '__main__':
    P = pd.read_csv(os.path.join(RES, 'primary_results.csv'))
    fig1()
    fig23('male', 'Fig2_sex_ratio_dense', 'Male share vs GEST (binned, FE-residualised)', 'Male share, residual (pp)',
          0.6, 'Male share, residual (pp)', P[P.model.str.startswith('H2: slopes')].set_index('term').b_pp.mean(), 'Country estimates')
    fig23('nnd', 'Fig3_mortality_dense', 'Neonatal death vs GEST (binned)', 'Neonatal death, residual (pp)',
          0.6, 'Neonatal death, residual (pp)', 0.0246, 'Country estimates')
