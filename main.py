import numpy as np
import xarray as xr
import rioxarray as riox
import os
import pandas as pd
from skimage.color import rgb2hsv
import yaml


def _threshold(path):
    with open(path, 'r') as yaml_file:
        parameters = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return parameters[0]


def _loader(path, chunks=(1, 2783, 2783)):

    path = os.path.normpath(path)
    filename = path.split(os.sep)[-1]
    date = pd.to_datetime(filename[7:15])
    band = filename[23:26]

    da = riox.open_rasterio(path, chunks=chunks)

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


def _ndvi(red, nir):

    ndvi = (nir - red) / (nir + red)
    ndvi.name = 'NDVI'

    return ndvi


def _evi(blue, red, nir):
    blue, red, nir = blue/1e4, red/1e4, nir/1e4

    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

    evi.name = 'EVI'

    return evi


def _hsv(mir, nir, red):

    rgb = np.dstack((mir[0], nir[0], red[0]))
    hsv = rgb2hsv(rgb)

    H = np.expand_dims(hsv[:, :, 0], 0)
    S = np.expand_dims(hsv[:, :, 1], 0)
    V = np.expand_dims(hsv[:, :, 2], 0)

    H = xr.DataArray(H,
                     dims=['time', 'y', 'x'],
                     coords={'time': mir.time, 'y': mir.y, 'x': mir.x})

    H = H.where(H != 0, np.NAN) * 360.

    S = xr.DataArray(S,
                     dims=['time', 'y', 'x'],
                     coords={'time': mir.time, 'y': mir.y, 'x': mir.x})

    V = xr.DataArray(V,
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

    alg = 'NDVI'
    parameters = 'parameters.yaml'
    limits = _threshold(parameters)

    ds = _composer(r'L:\HSL\observations\S2\scenes\2021\02\15\063\S2A_MSIL2A_20210215T065951_N0214_R063_T38NRL_20210215T100540.SAFE\GRANULE\L2A_T38NRL_A029518_20210215T070108\IMG_DATA\R20m')
    mask = _quality_mask(ds)
    ds_masked = ds.where(mask)

    ndvi = _ndvi(ds_masked.B04, ds_masked.B8A)
    evi = _evi(ds_masked.B02, ds_masked.B04, ds_masked.B8A)
    hsv = _hsv(ds_masked.B11, ds_masked.B8A, ds_masked.B04)

    if alg == 'EVI':
        combined = [evi, hsv.H]
    elif alg == 'NDVI':
        combined = [ndvi, hsv.H]

    gvi_ = _gvi(combined, **{'limits': limits})

    pass
