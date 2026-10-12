"""NASA POWER (MERRA-2, 0.5 x 0.625 deg) monthly temperature for all DHS regions, from the public AWS Zarr store.
No login needed. Output: data/power_monthly_anom.zarr with, per grid cell and month (1981 to latest available):
  T2M        monthly mean 2 m temperature (degC)
  T2M_MAX    maximum 2 m temperature within the month (degC)
  T2MWET     monthly mean wet-bulb temperature (degC)
  *_anom     departure from the cell's 1991-2020 calendar-month mean
The download reads the anonymous S3 bucket directly; env proxies are not used by s3fs/aiohttp.
"""
import os, time
import numpy as np, xarray as xr, s3fs

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, 'data'); os.makedirs(DATA, exist_ok=True)
STORE = 'nasa-power/merra2/temporal/power_merra2_monthly_temporal_lst.zarr'
VARS = ['T2M', 'T2M_MAX', 'T2MWET']
BBOX = dict(lat=slice(-36, 46), lon=slice(-120, 160))   # all DHS GPS countries (Latin America to Papua New Guinea)
YEARS = slice('1981-01-01', '2025-12-31')   # trailing months without data stay NaN
CLIM = slice('1991-01-01', '2020-12-31')


def download():
    raw = os.path.join(DATA, 'power_monthly_raw.zarr')
    if os.path.exists(raw): return xr.open_zarr(raw).load()
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={'region_name': 'us-west-2'})
    ds = xr.open_zarr(s3fs.S3Map(STORE, s3=fs), consolidated=True)[VARS].sel(time=YEARS, **BBOX)
    t = time.time(); ds = ds.load(); print('downloaded', dict(ds.sizes), round(time.time() - t), 's')
    for v in VARS: ds[v].attrs.update(units='degC')
    ds.to_zarr(raw, mode='w')
    return ds


def anomalies(ds):
    out = ds.copy()
    clim = ds.sel(time=CLIM).groupby('time.month').mean('time')
    for v in VARS:
        out[f'{v}_anom'] = ds[v].groupby('time.month') - clim[v]
        out[f'{v}_anom'].attrs.update(long_name=f'{v} minus 1991-2020 calendar-month mean', units='degC')
    return out.drop_vars('month', errors='ignore')


if __name__ == '__main__':
    ds = anomalies(download())
    last = int(ds.T2M.notnull().any(['lat', 'lon']).values.nonzero()[0].max())
    print('last month with data:', str(ds.time.values[last])[:7])
    ds.chunk({'time': -1, 'lat': 30, 'lon': 30}).to_zarr(os.path.join(DATA, 'power_monthly_anom.zarr'), mode='w')
    print('saved', os.path.join(DATA, 'power_monthly_anom.zarr'))
