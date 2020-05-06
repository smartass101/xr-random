import pytest
import numpy as np
import xarray as xr
from xrrandom.scipy_stats_sampling import sample_distribution, virtually_sample_distribution
from xrrandom.sampling import change_virtual_samples, _virtual_array


def test_virtual_array():
    arr = _virtual_array(1000)
    assert arr.shape == (1000,)


def test_returned_xarray():
    samples = sample_distribution('norm')
    assert isinstance(samples, xr.DataArray)
    assert samples.sizes == {'sample': 1}
    samples = sample_distribution('norm',
                                  loc=xr.Variable('means', [1, 2]))
    assert isinstance(samples, xr.DataArray)
    assert samples.ndim == 2


def test_sample_norm():
    loc = xr.DataArray(np.arange(5), dims=['loc'], name='loc')
    scale = xr.DataArray(np.arange(3)/2, dims=['scale'], name='scale')
    samples = sample_distribution('norm', 10, loc, scale=scale)
    assert samples.sizes == {'loc': 5, 'scale': 3, 'sample': 10}


def test_virtual_sample_norm():
    loc = xr.DataArray(np.arange(5), dims=['loc'], name='loc')
    scale = xr.DataArray(np.arange(3)/2, dims=['scale'], name='scale')
    samples = virtually_sample_distribution('norm', 10, loc, scale=scale)
    assert samples.sizes == {'loc': 5, 'scale': 3, 'sample': 10}
    samples_larger_dask = change_virtual_samples(samples, 100) * 2
    samples_larger = samples_larger_dask.compute()
    assert samples_larger.sizes['sample'] == 100



def test_virtual_sample_chunked_norm():
    loc = xr.DataArray(np.arange(5), dims=['loc'], name='loc')
    scale = xr.DataArray(np.arange(3)/2, dims=['scale'], name='scale')
    samples = virtually_sample_distribution('norm', 10, loc, scale=scale, sample_chunksize=5)
    assert samples.compute().sizes == {'loc': 5, 'scale': 3, 'sample': 10}
    with pytest.raises(ValueError):  # because sample dim is chunked
        samples_larger_dask = change_virtual_samples(samples, 100)
