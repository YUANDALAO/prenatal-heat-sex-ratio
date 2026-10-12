"""Link births to gestational temperature exposure from data/power_monthly_anom.zarr.

Input DataFrame columns: lat, lon, cmc_birth (DHS century-month code, b3: months since Jan 1900, Jan 1900 = 1).
Gestation is approximated as the 9 calendar months before the birth month:
  T1 = birth-9..birth-7, T2 = birth-6..birth-4, T3 = birth-3..birth-1, GEST = all nine.
Placebos: POST = birth..birth+11 (after sex is fixed), LEAD = birth+3..birth+11 (next year's same season).
Returns one column per <variable>_<window> (mean over the window's months).
"""
import os
import numpy as np, pandas as pd, xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = {'T1': (-9, -7), 'T2': (-6, -4), 'T3': (-3, -1), 'GEST': (-9, -1), 'POST': (0, 11), 'LEAD': (3, 11)}


def cmc_to_index(cmc, t0):
    """CMC -> position in the monthly time axis starting at t0 (a pandas Timestamp for Jan of first year)."""
    return (cmc - 1) - ((t0.year - 1900) * 12 + (t0.month - 1))


def link(births, variables=('T2M_anom', 'T2M_MAX_anom', 'T2MWET_anom'), path=None):
    ds = xr.open_zarr(path or os.path.join(HERE, 'data', 'power_monthly_anom.zarr'))
    t0 = pd.Timestamp(ds.time.values[0])
    lat_i = np.abs(ds.lat.values[None, :] - births.lat.values[:, None]).argmin(1)
    lon_i = np.abs(ds.lon.values[None, :] - births.lon.values[:, None]).argmin(1)
    base = cmc_to_index(births.cmc_birth.values.astype(int), t0)
    nt = ds.sizes['time']; out = pd.DataFrame(index=births.index)
    out['cell'] = lat_i * ds.sizes['lon'] + lon_i
    for v in variables:
        arr = ds[v].values                      # (time, lat, lon)
        for w, (a, b) in WINDOWS.items():
            idx = base[:, None] + np.arange(a, b + 1)[None, :]
            ok = (idx >= 0) & (idx < nt)
            vals = np.where(ok, arr[np.clip(idx, 0, nt - 1), lat_i[:, None], lon_i[:, None]], np.nan)
            out[f'{v}_{w}'] = np.where(ok.all(1), np.nanmean(vals, 1), np.nan)
    return out
