"""Figure 2 (main text): exposure-window profile in hot climates.
Each window is drawn across the months it covers (x = months relative to birth); height = estimate per 1 degC, band = 95% CI.
Pregnancy window in red; the pre-conception and post-birth windows (falsification / negative control) in grey.
"""
import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import figs_heat as fh

W = pd.read_csv(os.path.join(fh.RES, 'fig_windows.csv'))
SPAN = {'PRE': (-18, -10), 'T2M_anom_GEST': (-9, -1), 'T2M_anom_POST': (0, 11)}
SHORT = {'PRE': 'Before\nconception', 'T2M_anom_GEST': 'Pregnancy', 'T2M_anom_POST': 'After\nbirth'}
PANELS = [('male', 'A', 'Male share of births'), ('nnd', 'B', 'Neonatal death'), ('imr', 'C', 'Infant death')]


def f(x):
    v = f'{x:.2f}'
    return ('0.00' if v == '-0.00' else v).replace('-', '−').replace('.', '·')


def main():
    fig, axs = plt.subplots(1, 3, figsize=(180 * fh.MM, 68 * fh.MM), gridspec_kw={'wspace': 0.32})
    for ax, (out, tag, title) in zip(axs, PANELS):
        d = W[W.outcome == out].set_index('term')
        lo_all, hi_all = d.lo_pp.min(), d.hi_pp.max(); pad = 0.12 * (hi_all - lo_all)
        ax.axvspan(-9.5, -0.5, color='#f6f5f2', lw=0, zorder=0)
        ax.axhline(0, color=fh.GRAY, lw=0.8, zorder=1); ax.axvline(-0.5, color=fh.INK2, lw=0.7, ls=(0, (2, 2)), zorder=1)
        for k, (a, b) in SPAN.items():
            r = d.loc[k]; preg = k == 'T2M_anom_GEST'; col = fh.COL[2] if preg else fh.INK2
            x0, x1 = a - 0.45, b + 0.45
            ax.fill_between([x0, x1], r.lo_pp, r.hi_pp, color=col, alpha=0.18 if preg else 0.12, lw=0, zorder=2)
            ax.plot([x0, x1], [r.b_pp] * 2, color=col, lw=2.2, solid_capstyle='butt', zorder=3)
            ax.text((x0 + x1) / 2, r.hi_pp + pad * 0.2, f(r.b_pp), ha='center', va='bottom', fontsize=6.2,
                    color=col, fontweight='bold' if preg else 'normal')
            ax.text((x0 + x1) / 2, 0.985, SHORT[k], transform=ax.get_xaxis_transform(), ha='center', va='top',
                    fontsize=5.4, color=col, linespacing=1.0)
        ax.set_ylim(lo_all - pad, hi_all + pad * 2.6); ax.set_xlim(-19, 12)
        ax.set_xticks([-18, -9, 0, 11]); ax.set_xticklabels(['−18', '−9', 'Birth', '+11'])
        ax.set_xlabel('Months relative to birth'); ax.grid(True, axis='y'); ax.grid(False, axis='x')
        base = d.base.iloc[0]
        sub = (f'baseline {base * 100:.2f}% male' if out == 'male' else f'baseline {base * 1000:.1f} per 1000').replace('.', '·')
        ax.set_title(f'{title}\n', loc='left', fontweight='bold', fontsize=7)
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=6, color=fh.INK2, va='bottom')
        fh.tag(ax, tag)
        if out == 'male': ax.set_ylabel('Change per 1 °C warmer than usual\n(percentage points, 95% CI)')
    fh.save(fig, 'Fig2_windows')


if __name__ == '__main__':
    main()
