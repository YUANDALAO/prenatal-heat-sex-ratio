"""Pre-specified analyses H1-H3 and falsification (ANALYSIS_PLAN.md). Linear probability models with
cluster FE (survey x v001) and country x birth-year-month FE; SEs clustered by survey cluster.
Coefficients are reported in percentage points of male share per 1 degC anomaly.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)


def demean(cols, groups, df, tol=1e-9, max_iter=200):
    """Alternating projections onto multiple fixed effects (in place on a float64 matrix)."""
    M = df[cols].to_numpy(dtype=np.float64).copy()
    codes = [pd.factorize(df[g].to_numpy())[0] for g in groups]
    counts = [np.bincount(c) for c in codes]
    for it in range(max_iter):
        change = 0.0
        for c, n in zip(codes, counts):
            for j in range(M.shape[1]):
                m = np.bincount(c, weights=M[:, j]) / n
                M[:, j] -= m[c]; change = max(change, np.abs(m).max())
        if change < tol: break
    return M


def fe_ols(df, y, xs, fes, cluster):
    use = list(dict.fromkeys([y] + xs + fes + [cluster]))   # cluster may also be a fixed effect
    d = df[use].dropna()
    M = demean([y] + xs, fes, d)
    Y, X = M[:, 0], M[:, 1:]
    XtX_inv = np.linalg.inv(X.T @ X); b = XtX_inv @ X.T @ Y; e = Y - X @ b
    cl = pd.factorize(d[cluster].to_numpy())[0]; G = cl.max() + 1
    S = np.zeros((X.shape[1], X.shape[1]))
    sc = np.zeros((G, X.shape[1]))
    np.add.at(sc, cl, X * e[:, None])
    S = sc.T @ sc
    k = X.shape[1]; n = len(Y)
    V = XtX_inv @ S @ XtX_inv * (G / (G - 1))      # CR1 (fixed effects absorbed; small-sample factor on clusters)
    se = np.sqrt(np.diag(V))
    out = pd.DataFrame({'term': xs, 'b_pp': b * 100, 'se_pp': se * 100})
    out['lo_pp'] = out.b_pp - 1.96 * out.se_pp; out['hi_pp'] = out.b_pp + 1.96 * out.se_pp
    out['p'] = 2 * stats.norm.sf(np.abs(b / se)); out['n'] = n; out['clusters'] = G
    return out, V, b


def main():
    cols = ['survey', 'v001', 'cc', 'b3', 'male', 'primary', 'clim_tertile', 'bord',
            'T2M_anom_GEST', 'T2M_anom_POST', 'T2M_anom_LEAD', 'T2M_anom_T1', 'T2M_anom_T2', 'T2M_anom_T3',
            'T2M_MAX_anom_GEST', 'T2MWET_anom_GEST']
    B = pd.read_parquet(os.path.join(DATA, 'births.parquet'), columns=cols)
    B['clus'] = B.survey + '_' + B.v001.astype(int).astype(str)
    B['ctym'] = B.cc + '_' + B.b3.astype(int).astype(str)
    FE = ['clus', 'ctym']; R = []
    P = B[B.primary].copy()

    # H1 (primary): hot tertile, GEST with POST as negative control
    h1, _, _ = fe_ols(P[P.clim_tertile == 2], 'male', ['T2M_anom_GEST', 'T2M_anom_POST'], FE, 'clus')
    h1['model'] = 'H1 primary: hot tertile, GEST + POST'; R.append(h1)

    # H2: interaction by climate tertile (separate slopes), test hot - cool
    for t in range(3): P[f'G{t}'] = P.T2M_anom_GEST * (P.clim_tertile == t)
    for t in range(3): P[f'P{t}'] = P.T2M_anom_POST * (P.clim_tertile == t)
    xs = ['G0', 'G1', 'G2', 'P0', 'P1', 'P2']
    h2, V, b = fe_ols(P, 'male', xs, FE, 'clus')
    h2['model'] = 'H2: slopes by climate tertile (0 cool, 1 temperate, 2 hot)'; R.append(h2)
    d = b[2] - b[0]; se = np.sqrt(V[2, 2] + V[0, 0] - 2 * V[0, 2])
    R.append(pd.DataFrame([dict(model='H2 contrast', term='GEST hot - cool', b_pp=d * 100, se_pp=se * 100,
                                lo_pp=(d - 1.96 * se) * 100, hi_pp=(d + 1.96 * se) * 100, p=2 * stats.norm.sf(abs(d / se)),
                                n=h2.n.iloc[0], clusters=h2.clusters.iloc[0])]))

    # H3: bins of GEST within each tertile (reference -0.5..0.5)
    bins = [-np.inf, -1, -0.5, 0.5, 1, np.inf]; labs = ['lt_m1', 'm1_m05', 'ref', 'p05_p1', 'gt_p1']
    P['gbin'] = pd.cut(P.T2M_anom_GEST, bins, labels=labs)
    for t in range(3):
        Q = P[P.clim_tertile == t].copy()
        bx = []
        for l in labs:
            if l == 'ref': continue
            Q[f'bin_{l}'] = (Q.gbin == l).astype(float); bx.append(f'bin_{l}')
        h3, _, _ = fe_ols(Q, 'male', bx + ['T2M_anom_POST'], FE, 'clus')
        h3['model'] = f'H3 bins, tertile {t}'; R.append(h3)

    # Falsification within hot tertile: LEAD window; pre-conception window not built yet -> reported later
    H = P[P.clim_tertile == 2]
    f1, _, _ = fe_ols(H, 'male', ['T2M_anom_GEST', 'T2M_anom_LEAD'], FE, 'clus'); f1['model'] = 'Falsification: GEST + LEAD (hot)'; R.append(f1)
    # H6 exploratory: trimesters jointly (hot)
    h6, _, _ = fe_ols(H, 'male', ['T2M_anom_T1', 'T2M_anom_T2', 'T2M_anom_T3', 'T2M_anom_POST'], FE, 'clus'); h6['model'] = 'H6 trimesters (hot)'; R.append(h6)
    # Secondary exposures (hot)
    for v in ['T2M_MAX_anom_GEST', 'T2MWET_anom_GEST']:
        s, _, _ = fe_ols(H, 'male', [v], FE, 'clus'); s['model'] = f'Secondary exposure {v} (hot)'; R.append(s)
    # Secondary sample: India and Nepal first births, all tertiles pooled by tertile slopes
    S = B[B.cc.isin(['IA', 'NP']) & (B.bord == 1)].copy()
    for t in range(3): S[f'G{t}'] = S.T2M_anom_GEST * (S.clim_tertile == t)
    s2, _, _ = fe_ols(S, 'male', ['G0', 'G1', 'G2', 'T2M_anom_POST'], FE, 'clus'); s2['model'] = 'Secondary: India+Nepal first births'; R.append(s2)

    O = pd.concat(R, ignore_index=True)
    O.to_csv(os.path.join(RES, 'primary_results.csv'), index=False)
    pd.set_option('display.width', 250); print(O[['model', 'term', 'b_pp', 'lo_pp', 'hi_pp', 'p', 'n', 'clusters']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
