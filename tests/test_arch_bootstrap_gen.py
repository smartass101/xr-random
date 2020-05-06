import numpy as np
import xarray as xr
from xrrandom import bootstrap_samples

def test_bootstrap_samples():
    data = xr.DataArray(np.empty((500, 100)), dims=['time', 'x'])
    bs = bootstrap_samples(data, 'time', 1000)
    assert bs.sizes['sample'] == 1000
    assert bs.ndim == 3
    repr(bs)                    # this should not fail

def test_simple_1d_bootstrap():
    t = np.linspace(0, 10, 100)
    data = xr.DataArray(np.sin(t), coords=[('time', t)], name='sig')
    resampled = bootstrap_samples(data, 'time', 1000)  # 1000 IIDBootstrap samples by default
    means = resampled.mean(dim='time')
    mean = means.mean(dim='sample')
    mean_unc = means.std(dim='sample')
