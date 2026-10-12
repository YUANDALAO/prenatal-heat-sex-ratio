"""Pregnancy outcomes for H4 from DHS Women's Recode (IR), linked to temperature in the 9 months before the outcome.

DHS-4..7: reproductive calendar vcal_1. Position i (0-based) is CMC v017 + len - 1 - i; validated 100% against
          b3_01 in three surveys. 'B' = live birth month, 'T' = termination month (miscarriage, stillbirth or
          induced abortion, not distinguished).
DHS-8:    pregnancy history p3_xx (CMC of outcome), p32_xx (1 live birth, 2 stillbirth, 3 miscarriage, 4 abortion),
          p4_xx sex, p20_xx duration (months).
Window:   outcomes in the 60 months before interview (v008-60 .. v008-1), the calendar's common span.
Calendar shifts for Ethiopia / Nepal are taken from data/survey_qc.csv (same as births).
Output:   data/pregnancies.parquet (no coordinates).
"""
import io, os, re, zipfile
import numpy as np, pandas as pd
from build_births import RAW, DATA, file_to_survey, read_ge, SEX_SELECTION
from exposure import link


def read_ir(zname):
    with zipfile.ZipFile(os.path.join(RAW, zname)) as z:
        n = [i for i in z.namelist() if i.lower().endswith('.dta')][0]; raw = z.read(n)
    cols = list(pd.read_stata(io.BytesIO(raw), iterator=True, convert_categoricals=False).variable_labels())
    base = [c for c in ['v001', 'v005', 'v008', 'v017'] if c in cols]
    preg = [c for c in cols if re.match(r'^(p3|p32|p4|p20)_\d\d$', c)]
    if preg:
        return 'dhs8', pd.read_stata(io.BytesIO(raw), columns=base + preg, convert_categoricals=False)
    if 'vcal_1' in cols:
        return 'calendar', pd.read_stata(io.BytesIO(raw), columns=base + ['vcal_1'], convert_categoricals=False)
    return None, None


def events_calendar(d):
    out = []
    for v001, v008, v017, s in zip(d.v001, d.v008, d.v017, d.vcal_1.astype(str)):
        L = len(s)
        for i, ch in enumerate(s):
            if ch in 'BT':
                cm = v017 + L - 1 - i
                if v008 - 60 <= cm <= v008 - 1: out.append((v001, cm, int(ch == 'T'), np.nan, np.nan, np.nan))
    return pd.DataFrame(out, columns=['v001', 'cmc_end', 'term', 'p32', 'sex', 'dur'])


def events_dhs8(d):
    k = sorted({c.split('_')[1] for c in d.columns if c.startswith('p3_')})
    rows = []
    for j in k:
        e = pd.DataFrame({'v001': d.v001, 'v008': d.v008, 'cmc_end': d[f'p3_{j}'], 'p32': d.get(f'p32_{j}'),
                          'sex': d.get(f'p4_{j}'), 'dur': d.get(f'p20_{j}')})
        rows.append(e[e.cmc_end.notna()])
    e = pd.concat(rows)
    e = e[(e.cmc_end >= e.v008 - 60) & (e.cmc_end <= e.v008 - 1) & e.p32.isin([1, 2, 3, 4])]
    e['term'] = (e.p32 != 1).astype(int)
    return e.drop(columns='v008')


def main():
    f2s = file_to_survey(); qc = pd.read_csv(os.path.join(DATA, 'survey_qc.csv'), dtype={'surv_id': str}).set_index('surv_id')
    ge_by_sid = {s: f for f, sids in f2s.items() if f[2:4] == 'GE' and os.path.exists(os.path.join(RAW, f)) for s in sids}
    parts, log = [], []
    for f in sorted(x for x in os.listdir(RAW) if re.match(r'^[A-Z]{2}IR..DT\.ZIP$', x)):
        sids = [s for s in f2s.get(f, []) if s in ge_by_sid and s in qc.index]
        if not sids: log.append(dict(file=f, status='no GPS / not in births QC')); continue
        sid = sids[0]; shift = int(qc.loc[sid, 'cmc_shift_applied'])
        kind, d = read_ir(f)
        if kind is None: log.append(dict(file=f, status='no calendar or pregnancy history')); continue
        for c in ['v008', 'v017'] + [c for c in d.columns if c.startswith('p3_')]:
            if c in d: d[c] = d[c] + shift
        e = events_calendar(d) if kind == 'calendar' else events_dhs8(d)
        e = e.merge(read_ge(ge_by_sid[sid]), on='v001', how='inner')
        e['survey'] = f[:8]; e['cc'] = f[:2]; e['source'] = kind
        parts.append(e); log.append(dict(file=f, surv_id=sid, source=kind, events=len(e), terminations=int(e.term.sum()), status='ok'))
        print(f, kind, len(e), int(e.term.sum()), flush=True)
    E = pd.concat(parts, ignore_index=True)
    E = E[E.cmc_end >= (1982 - 1900) * 12 + 10].copy()
    E['primary'] = ~E.cc.isin(SEX_SELECTION)
    X = link(E.rename(columns={'cmc_end': 'cmc_birth'})[['lat', 'lon', 'cmc_birth']], variables=('T2M_anom',))
    E = E.join(X).drop(columns=['lat', 'lon'])
    import xarray as xr, json
    ds = xr.open_zarr(os.path.join(DATA, 'power_monthly_anom.zarr'))
    clim = ds.T2M.sel(time=slice('1991-01-01', '2020-12-31')).mean('time').values.ravel()
    cuts = json.load(open(os.path.join(DATA, 'climate_tertiles.json')))['tertile_cuts_degC']   # same cut points as births
    E['clim_tertile'] = np.digitize(clim[E.cell.values], cuts)
    E.to_parquet(os.path.join(DATA, 'pregnancies.parquet'), index=False)
    pd.DataFrame(log).to_csv(os.path.join(DATA, 'pregnancy_qc.csv'), index=False)
    print('events', len(E), 'terminations', int(E.term.sum()), 'surveys', E.survey.nunique())


if __name__ == '__main__':
    main()
