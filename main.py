import numpy as np
import xarray as xr
import rioxarray as riox
from rioxarray.merge import merge_arrays
import os
import pandas as pd
from skimage.color import rgb2hsv
import yaml
import matplotlib.pyplot as plt
import datetime
from distributed import Client, progress
from numba import njit


def _decades(data):
    if data[-1] == 1:
        diff = np.diff(data)
        # return np.split(data, np.where(np.diff(data) != 0)[0]+1)[-1].size
        return np.split(data, np.where(np.logical_and(~np.isnan(diff), diff != 0))[0]+1)[-1].size
    else:
        return -999


def _folder_walker(scene_path):
    orbits = ['006', '020', '035', '049', '063', '078', '092', '106', '135']

    Q = [str(i) + 'Q' for i in range(37, 40)]
    P = [str(i) + 'P' for i in range(36, 40)]
    N = [str(i) + 'N' for i in range(36, 40)]
    M = [str(i) + 'M' for i in range(36, 39)]

    zones = []
    for i in (Q, P, N, M):
        zones.extend(i)

    path = os.path.normpath(scene_path)
    tiles = []

    datelist = pd.date_range(end=datetime.datetime.today(), periods=120, freq='D').tolist()

    for obs_date in datelist:
        year = str(obs_date.year)
        month = str(obs_date.month).zfill(2)
        day = str(obs_date.day).zfill(2)

        path_date = os.path.join(path, year, month, day)

        # tile_date = []
        if os.path.isdir(path_date):
            dic_zones = {zn: [] for zn in range(36, 40)}
            for orbit in orbits:

                path_orbit = os.path.join(path_date, orbit)

                if os.path.isdir(path_orbit):
                    for folder in os.listdir(path_orbit):
                        zone = folder[39:42]
                        if zone in zones:

                            path_granule = os.path.join(path_orbit, folder, 'GRANULE')
                            for observation in os.listdir(path_granule):
                                bands_path = os.path.join(path_granule, observation, 'IMG_data', 'R20m')
                                upd_lst_int = dic_zones.get(int(zone[0:2]))
                                upd_lst_int.append(bands_path)
                                dic_zones.update({int(zone[0:2]): upd_lst_int})
                                # tile_date.append(bands_path)
            # tiles.append(tile_date)
            tiles.append(list(filter(None, dic_zones.values())))

    return tiles


def _threshold(path):
    with open(path, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return parameters[0]


def _loader(path, ):

    path = os.path.normpath(path)
    filename = path.split(os.sep)[-1]
    date = pd.to_datetime(filename[7:15])
    band = filename[23:26]

    da = riox.open_rasterio(path)  # TODO add chunks

    da = da.assign_coords({'time': ('band', [date, ])})
    da = da.swap_dims({'band': 'time'})
    da = da.drop_vars('band')
    da.name = band

    return (da.name, da)


def _composer(tile_path):
    for root, dirs, files in os.walk(tile_path):
        tile = []
        bands_nms = ['B02', 'B03', 'B04', 'B8A', 'B11', 'SCL']
        bands_da = []

        for filename in files:
            for i in bands_nms:
                if i in filename and filename.endswith('.jp2'):
                    filepath = os.path.join(root, filename)
                    bands_da.append(_loader(filepath))

        das = dict(bands_da)

        return xr.Dataset(das)


def _quality_mask(ds):
    return ds.SCL.where(((ds.SCL != 0) &
                         (ds.SCL != 1) &
                         (ds.SCL != 2) &
                         (ds.SCL != 6) &
                         (ds.SCL != 11) &
                         (ds.SCL != 8) &
                         (ds.SCL != 9))).notnull()


@njit(cache=True)
def __ndvicpu(R, NIR, scale=10000.):
    NDVI_ = np.zeros(R.shape, dtype=np.float64)
    rows, cols = R.shape

    for y in range(0, rows):
        for x in range(0, cols):
            R_ = R[y, x]/scale
            NIR_ = NIR[y, x]/scale
            denominator = (NIR_ + R_)

            if not denominator == 0 or denominator == np.NAN:
                ndvi_ = (NIR_ - R_) / (NIR_ + R_)
                NDVI_[y, x] = ndvi_
            else:
                NDVI_[y, x] = np.NAN



    return NDVI_


def _ndvi(red, nir):

    ndvi_arr = __ndvicpu(red.values[0], nir.values[0])

    ndvi_arr = np.expand_dims(ndvi_arr, 0)

    ndvi_da = xr.DataArray(ndvi_arr,
                           dims=['time', 'y', 'x'],
                           coords={'time': nir.time, 'y': nir.y, 'x': nir.x, 'spatial_ref': nir.spatial_ref},
                           attrs={'scale_factor': nir.scale_factor,
                                  'add_offset': nir.add_offset,
                                  'grid_mapping': nir.grid_mapping})

    ndvi_da.name = 'NDVI'

    return ndvi_da


@njit(cache=True)
def __evicpu(blue, red, nir):
    blue, red, nir = blue/1e4, red/1e4, nir/1e4

    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

    evi.name = 'EVI'

    return evi


def _evi(blue, red, nir):
    evi_arr = __evicpu(blue.values[0], red.values[0], nir.values[0])

    evi_arr = np.expand_dims(evi_arr, 0)

    evi_da = xr.DataArray(evi_arr,
                          dims=['time', 'y', 'x'],
                          coords={'time': nir.time, 'y': nir.y, 'x': nir.x, 'spatial_ref': nir.spatial_ref},
                          attrs={'scale_factor': nir.scale_factor,
                                 'add_offset': nir.add_offset,
                                 'grid_mapping': nir.grid_mapping})

    evi_da.name = 'EVI'

    return  evi_da


@njit(cache=True)
def _rgb2hsvcpu(R, G, B, scale=10000.):
    """
    RGB to HSV
    Convert RGB values to HSV taking into account negative values
    Adapted from https://stackoverflow.com/questions/39118528/rgb-to-hsl-conversion

    :param
    R_ : float
      red channel
    G_ : float
      green channel
    B_ : float
      blue channel
    scale: float, optional

    :returns
    H, S, V : float

    """
    H = np.zeros(R.shape, dtype=np.float64)
    S = np.zeros(R.shape, dtype=np.float64)
    V = np.zeros(R.shape, dtype=np.float64)

    rows, cols = R.shape
    for y in range(0, rows):
        for x in range(0, cols):
            h, s, v = 0, 0, 0

            R_ = R[y, x]/scale
            G_ = G[y, x]/scale
            B_ = B[y, x]/scale

            Cmax = max(R_, G_, B_)
            Cmin = min(R_, G_, B_)
            croma = Cmax - Cmin

            if croma == 0:
                H[y, x] = np.NAN
                S[y, x] = np.NAN
                V[y, x] = np.NAN
                continue

            if Cmax == R_:
                segment = (G_ - B_ ) / croma
                shift = 0 / 60
                if segment < 0:
                    shift = 360 / 60
                h = segment + shift

            if Cmax == G_:
                 segment = (B_ - R_) / croma
                 shift   = 120 / 60
                 h = segment + shift

            if Cmax == B_:
                segment = (R_ - G_) / croma
                shift   = 240 / 60
                h = segment + shift

            h *= 60.
            v = Cmax
            s = croma/v

            H[y, x] = h
            S[y, x] = s*100.
            V[y, x] = v* 100.

    return H, S, V


def _hsv(mir, nir, red):

    h_arr, s_arr, v_arr = _rgb2hsvcpu(mir.values[0], nir.values[0], red.values[0])
    h_arr = np.expand_dims(h_arr, 0)
    s_arr = np.expand_dims(s_arr, 0)
    v_arr = np.expand_dims(v_arr, 0)

    H = xr.DataArray(h_arr,
                     dims=['time', 'y', 'x'],
                     coords={'time': mir.time, 'y': mir.y, 'x': mir.x})

    H = H.where(H != 0, np.NAN) * 360.

    S = xr.DataArray(s_arr,
                     dims=['time', 'y', 'x'],
                     coords={'time': mir.time, 'y': mir.y, 'x': mir.x})

    V = xr.DataArray(v_arr,
                     dims=['time', 'y', 'x'],
                     coords={'time': mir.time, 'y': mir.y, 'x': mir.x})

    HSV = xr.Dataset({'H': H, 'S': S, 'V': V})

    return HSV


def _gvi(combined, **parameters):

    limits = parameters.pop('limits', '')

    V_INDEX, H = combined

    null_mask = np.logical_or(V_INDEX.isnull(), H.isnull())

    vegetated = H.where(eval(limits['equation']))

    gvi = vegetated.notnull()

    gvi = gvi.where(~null_mask)

    gvi.name = 'GVI'

    return gvi


if __name__ == '__main__':

    client = Client()

    sensor = 'A'
    decad = '20210221'
    alg = 'NDVI'
    parameters = 'parameters.yaml'
    limits = _threshold(parameters)

    tiles_pth = _folder_walker(r'L:\HSL\observations\S2\sub')

    dates_da = []

    for agg_date in tiles_pth:
        for agg_utm in agg_date:
            tiles = []

            for sng_obs_tile in agg_utm:
                ds = _composer(sng_obs_tile)
                mask = _quality_mask(ds)
                ds_masked = ds.where(mask)

                hsv = _hsv(ds_masked.B11, ds_masked.B8A, ds_masked.B04)

                if alg == 'EVI':
                    evi = _evi(ds_masked.B02, ds_masked.B04, ds_masked.B8A)
                    combined = [evi, hsv.H]
                elif alg == 'NDVI':
                    ndvi = _ndvi(ds_masked.B04, ds_masked.B8A)
                    combined = [ndvi, hsv.H]

                gvi_ = _gvi(combined, **{'limits': limits})

                tiles.append(gvi_)

            sng_date_utm_strip = merge_arrays(tiles, method='max', nodata=np.NAN)

            sng_date_reproj = sng_date_utm_strip.rio.reproject('EPSG:4326', )

            dates_da.append(sng_date_reproj)

    # time_agg = xr.combine_nested(dates_da, concat_dim=['time'])
    time_agg = xr.combine_nested(dates_da, concat_dim=[['time', 'x', 'y']])

    GVDM = xr.apply_ufunc(_decades, time_agg,
                          input_core_dims=[['time']],
                          exclude_dims={'time', },
                          dask='parallelized',
                          dask_gufunc_kwargs={'allow_rechunk': True},
                          vectorize=True,)
    GVDM.name = 'GVI'

    GVDM_f = GVDM.to_netcdf(rf'L:\HSL\observations\S2\GVI_S2{sensor}_{decad}_{alg}.nc',
                            compute=False,
                            encoding={'GVI': {'_FillValue': -999}}).persist()

    progress(GVDM_f)

    client.close()


    pass
