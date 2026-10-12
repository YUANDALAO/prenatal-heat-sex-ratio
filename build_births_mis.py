"""Replication sample (ANALYSIS_PLAN.md deviation 9): births from MIS/AIS surveys with GPS.
BR where available, otherwise the IR women's file reshaped from wide (b*_01..b*_20) to one row per birth.
Same rules as build_births.py, but births 1-60 months before interview. Output: data/births_mis_ais.parquet.
"""
import io, json, os, re, zipfile
import numpy as np, pandas as pd
import build_births as bb
from exposure import link

RAW = os.path.join(bb.RAW, 'mis_ais')
bb.RAW = RAW                                      # read_ge / read_br use bb.RAW
B_VARS = ['b0', 'b3', 'b4', 'b5', 'b7', 'b11']
W_VARS = ['v000', 'v001', 'v002', 'v003', 'v005', 'v007', 'v008', 'v012', 'v024', 'v025', 'v104', 'v106']


def file_to_survey():
    m = {}
    for u in open(os.path.join(bb.HERE, 'dhs_raw', 'urls_mis_ais_needed.txt')):
        if 'Filename=' in u:
            m.setdefault(re.search(r'Filename=([A-Za-z0-9]+\.(?:ZIP|zip))', u).group(1).upper(), []).append(
                re.search(r'surv_id=(\d+)', u).group(1))
    return m


def read_ir_births(zname):
    with zipfile.ZipFile(os.path.join(RAW, zname)) as z:
        n = [i for i in z.namelist() if i.lower().endswith('.dta')][0]; raw = z.read(n)
    cols = list(pd.read_stata(io.BytesIO(raw), iterator=True, convert_categoricals=False).variable_labels())
    idx = sorted({c.rsplit('_', 1)[1] for c in cols if re.match(r'^b3_\d{1,2}$', c)}, key=int)   # b3_01 or b3_1
    if not idx: return None                                    # no birth history in this IR (e.g. some AIS)
    use = [c for c in W_VARS if c in cols] + [f'{v}_{i}' for v in B_VARS for i in idx if f'{v}_{i}' in cols]
    d = pd.read_stata(io.BytesIO(raw), columns=use, convert_categoricals=False)
    parts = []
    for i in idx:
        p = d[[c for c in W_VARS if c in d]].copy()
        for v in B_VARS: p[v] = d[f'{v}_{i}'] if f'{v}_{i}' in d else np.nan
        p['bidx'] = int(i); parts.append(p[p.b3.notna()])
    if not parts: return None
    return pd.concat(parts, ignore_index=True)


def main():
    f2s = file_to_survey()
    API = pd.read_csv(os.path.join(bb.DATA, 'dhs_api_surveys_mis_ais.csv')); API['SurveyNum'] = API.SurveyNum.astype(str)
    for c in ['FieldworkStart', 'FieldworkEnd']: API[c] = pd.to_datetime(API[c], errors='coerce')
    API = API.set_index('SurveyNum')
    ge = {s: f for f, ss in f2s.items() if f[2:4] == 'GE' for s in ss}
    br = {s: f for f, ss in f2s.items() if f[2:4] == 'BR' for s in ss}
    ir = {s: f for f, ss in f2s.items() if f[2:4] == 'IR' for s in ss}
    parts, qc = [], []
    for sid, gfile in sorted(ge.items()):
        src = 'BR' if sid in br else 'IR'
        b = bb.read_br(br[sid]) if src == 'BR' else read_ir_births(ir[sid])
        if b is None:
            qc.append(dict(surv_id=sid, source=src, ge=gfile, status='no birth history in IR; no BR')); print(sid, 'no births'); continue
        meta = API.loc[sid] if sid in API.index else None
        mid = meta.FieldworkStart + (meta.FieldworkEnd - meta.FieldworkStart) / 2 if meta is not None and pd.notna(meta.FieldworkStart) and pd.notna(meta.FieldworkEnd) else None
        offset = int(round(bb.cmc(mid) - b.v008.median())) if (mid is not None and pd.notna(b.v008.median())) else 0
        if mid is None: print(sid, 'no fieldwork dates in API; calendar check skipped; v008 years', sorted(((b.v008.dropna() - 1) // 12 + 1900).astype(int).unique())[:3])
        shift = offset if abs(offset) >= 24 else 0
        cc = gfile[:2]
        if shift and cc in bb.KNOWN_OFFSETS and abs(shift - bb.KNOWN_OFFSETS[cc]) <= 3: shift = bb.KNOWN_OFFSETS[cc]
        for c in ['v008', 'b3']: b[c] = b[c] + shift
        g = bb.read_ge(gfile); b = b.merge(g, on='v001', how='inner')
        b['survey'] = gfile[:2] + src + gfile[4:8]; b['surv_id'] = sid; b['cc'] = cc
        b['survey_type'] = None if meta is None else meta.SurveyType
        qc.append(dict(surv_id=sid, survey_id=None if meta is None else meta.SurveyId, source=src, ge=gfile,
                       births_raw=len(b), cmc_offset_found=offset, cmc_shift_applied=shift))
        parts.append(b); print(sid, src, gfile, len(b), 'shift', shift, flush=True)
    B = pd.concat(parts, ignore_index=True)
    B['age_m'] = B.v008 - B.b3
    B = B[(B.age_m >= 1) & (B.age_m <= 60) & (B.b0 == 0) & (B.b3 >= (1982 - 1900) * 12 + 1) & B.b4.isin([1, 2])].copy()
    B['male'] = (B.b4 == 1).astype('int8'); B['primary'] = ~B.cc.isin(bb.SEX_SELECTION)
    E = link(B.rename(columns={'b3': 'cmc_birth'})[['lat', 'lon', 'cmc_birth']]); B = B.join(E)
    import xarray as xr
    ds = xr.open_zarr(os.path.join(bb.DATA, 'power_monthly_anom.zarr'))
    clim = ds.T2M.sel(time=slice('1991-01-01', '2020-12-31')).mean('time').values.ravel()
    cuts = json.load(open(os.path.join(bb.DATA, 'climate_tertiles.json')))['tertile_cuts_degC']   # primary-sample cut points
    B['clim_mean'] = clim[B.cell.values]; B['clim_tertile'] = np.digitize(B.clim_mean, cuts)
    B = B.drop(columns=['lat', 'lon'])
    B.to_parquet(os.path.join(bb.DATA, 'births_mis_ais.parquet'), index=False)
    pd.DataFrame(qc).to_csv(os.path.join(bb.DATA, 'survey_qc_mis_ais.csv'), index=False)
    print('births', len(B), 'surveys', B.surv_id.nunique(), 'primary', int(B.primary.sum()),
          'hot primary', int((B.primary & (B.clim_tertile == 2)).sum()))


if __name__ == '__main__':
    main()
