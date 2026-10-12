"""Build the analysis file of births from DHS Births Recode (BR) + GPS (GE), linked to gestational temperature.

Inputs: dhs_raw/*.ZIP (renamed by the download step), dhs_raw/urls_needed.txt (file -> surv_id),
DHS API survey metadata (fieldwork dates), data/power_monthly_anom.zarr.
Output: data/births.parquet (one row per birth; no coordinates kept), data/survey_qc.csv.

Calendar check: some surveys record dates in a national calendar (e.g. Ethiopia, Nepal). For every survey, the
median interview CMC (v008) is compared with the fieldwork midpoint from the DHS API; a survey whose offset is at least
24 months (calendar differences are ~92 months for Ethiopia, ~681 for Nepal; pooled continuous surveys such as Peru
2003-08 differ from the API window by <24) is shifted by that offset. All offsets are logged in survey_qc.csv.
"""
import io, json, os, re, struct, sys, urllib.request, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, 'dhs_raw'); DATA = os.path.join(HERE, 'data')
sys.path.insert(0, HERE)
from exposure import link

BR_COLS = ['v000', 'v001', 'v002', 'v003', 'v005', 'v007', 'v008', 'v012', 'v024', 'v025', 'v104', 'v106',
           'bidx', 'bord', 'b0', 'b3', 'b4', 'b5', 'b7', 'b11', 'b20']
KNOWN_OFFSETS = {'ET': 92, 'NP': -681}   # Ethiopian and Nepali (Bikram Sambat) calendars vs Gregorian CMC
SEX_SELECTION = {'IA', 'CN', 'VN', 'AM', 'AZ', 'AL', 'PK', 'NP'}   # DHS country codes; excluded from primary analysis


def file_to_survey():
    m = {}
    for u in open(os.path.join(RAW, 'urls_needed.txt')):
        if 'Filename=' not in u: continue
        fn = re.search(r'Filename=([A-Za-z0-9]+\.(?:ZIP|zip))', u).group(1).upper()
        m.setdefault(fn, []).append(re.search(r'surv_id=(\d+)', u).group(1))
    return m


def api_surveys():
    """DHS API survey metadata; cached in data/dhs_api_surveys.csv (fetched 2026-09-30) to avoid network failures."""
    cache = os.path.join(DATA, 'dhs_api_surveys.csv')
    if os.path.exists(cache):
        S = pd.read_csv(cache)
    else:
        for attempt in range(5):
            try:
                d = json.load(urllib.request.urlopen('https://api.dhsprogram.com/rest/dhs/surveys?f=json&perpage=2000', timeout=120))['Data']
                break
            except Exception:
                if attempt == 4: raise
        S = pd.DataFrame(d); S.to_csv(cache, index=False)
    S['SurveyNum'] = S.SurveyNum.astype(str)
    for c in ['FieldworkStart', 'FieldworkEnd']: S[c] = pd.to_datetime(S[c], errors='coerce')
    return S.set_index('SurveyNum')


def read_dbf(data):
    nrec, hlen, rlen = struct.unpack('<IHH', data[4:12]); fields = []; p = 32
    while data[p] != 0x0D:
        fields.append((data[p:p + 11].split(b'\0')[0].decode(), data[p + 16])); p += 32
    rows = []
    for r in range(nrec):
        rec = data[hlen + r * rlen: hlen + (r + 1) * rlen]; o = 1; row = {}
        for name, ln in fields:
            row[name] = rec[o:o + ln].decode('latin1').strip('\x00 ').strip(); o += ln
        rows.append(row)
    return pd.DataFrame(rows)


def read_ge(zname):
    with zipfile.ZipFile(os.path.join(RAW, zname)) as z:
        n = [i for i in z.namelist() if i.lower().endswith('.dbf')][0]; g = read_dbf(z.read(n))
    g = g.rename(columns=str.upper)
    out = pd.DataFrame({'v001': pd.to_numeric(g.DHSCLUST, errors='coerce'),
                        'lat': pd.to_numeric(g.LATNUM, errors='coerce'), 'lon': pd.to_numeric(g.LONGNUM, errors='coerce')})
    bad = (out.lat.abs() < 1e-6) & (out.lon.abs() < 1e-6)
    if 'SOURCE' in g: bad |= g.SOURCE.str.upper().eq('MIS')
    return out[~bad & out.lat.notna()].drop_duplicates('v001')


def read_br(zname):
    with zipfile.ZipFile(os.path.join(RAW, zname)) as z:
        n = [i for i in z.namelist() if i.lower().endswith('.dta')][0]
        raw = z.read(n)
    it = pd.read_stata(io.BytesIO(raw), iterator=True, convert_categoricals=False)
    cols = [c for c in BR_COLS if c in it.variable_labels()]
    return pd.read_stata(io.BytesIO(raw), columns=cols, convert_categoricals=False)


def cmc(ts): return (ts.year - 1900) * 12 + ts.month


def main():
    f2s = file_to_survey(); API = api_surveys()
    br_files = sorted(f for f in os.listdir(RAW) if re.match(r'^[A-Z]{2}BR..DT\.ZIP$', f))
    ge_by_sid = {s: f for f, sids in f2s.items() if f[2:4] == 'GE' and os.path.exists(os.path.join(RAW, f)) for s in sids}
    qc, parts = [], []
    for f in br_files:
        sids = [s for s in f2s.get(f, []) if s in ge_by_sid]
        if not sids: qc.append(dict(file=f, status='no GPS file')); continue
        sid = sids[0]; ge = read_ge(ge_by_sid[sid]); b = read_br(f)
        meta = API.loc[sid] if sid in API.index else None
        mid = meta.FieldworkStart + (meta.FieldworkEnd - meta.FieldworkStart) / 2 if meta is not None and pd.notna(meta.FieldworkStart) else None
        offset = int(round(cmc(mid) - b.v008.median())) if mid is not None else 0
        shift = offset if abs(offset) >= 24 else 0
        if shift and f[:2] in KNOWN_OFFSETS and abs(shift - KNOWN_OFFSETS[f[:2]]) <= 3:
            shift = KNOWN_OFFSETS[f[:2]]     # snap to the documented calendar offset (estimate is +-1 month)
        for c in ['v008', 'b3']: b[c] = b[c] + shift
        b = b.merge(ge, on='v001', how='inner')
        b['survey'] = f[:8]; b['surv_id'] = sid; b['cc'] = f[:2]
        qc.append(dict(file=f, surv_id=sid, survey_id=None if meta is None else meta.SurveyId, ge=ge_by_sid[sid],
                       births_raw=len(b), clusters_gps=ge.v001.nunique(), cmc_offset_found=offset, cmc_shift_applied=shift,
                       status='ok'))
        parts.append(b); print(f, sid, len(b), 'shift', shift, flush=True)
    B = pd.concat(parts, ignore_index=True)
    # sample definition (ANALYSIS_PLAN.md)
    B['age_m'] = B.v008 - B.b3
    B = B[(B.age_m >= 1) & (B.age_m <= 180) & (B.b0 == 0) & (B.b3 >= (1982 - 1900) * 12 + 1) & B.b4.isin([1, 2])].copy()
    B['male'] = (B.b4 == 1).astype('int8')
    B['primary'] = ~B.cc.isin(SEX_SELECTION)
    E = link(B.rename(columns={'b3': 'cmc_birth'})[['lat', 'lon', 'cmc_birth']])
    B = B.join(E)
    # baseline climate of the grid cell: mean annual T2M 1991-2020; tertile cut points fixed here, before outcomes
    import xarray as xr
    ds = xr.open_zarr(os.path.join(DATA, 'power_monthly_anom.zarr'))
    clim = ds.T2M.sel(time=slice('1991-01-01', '2020-12-31')).mean('time').values.ravel()
    B['clim_mean'] = clim[B.cell.values]
    cl = B.drop_duplicates(['survey', 'v001'])          # cluster-weighted distribution, as in the plan
    cuts = np.nanpercentile(cl.clim_mean, [100 / 3, 200 / 3])
    B['clim_tertile'] = np.digitize(B.clim_mean, cuts)  # 0 cool, 1 temperate, 2 hot
    json.dump({'tertile_cuts_degC': [float(c) for c in cuts], 'clusters': int(len(cl))},
              open(os.path.join(DATA, 'climate_tertiles.json'), 'w'), indent=1)
    B = B.drop(columns=['lat', 'lon'])        # coordinates are not stored in the analysis file
    os.makedirs(DATA, exist_ok=True)
    B.to_parquet(os.path.join(DATA, 'births.parquet'), index=False)
    pd.DataFrame(qc).to_csv(os.path.join(DATA, 'survey_qc.csv'), index=False)
    print('births', len(B), 'surveys', B.survey.nunique(), 'primary', int(B.primary.sum()))


if __name__ == '__main__':
    main()
