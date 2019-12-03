import numpy as np
import xarray as xr
from xrrandom import bootstrap_samples

def test_bootstrap_samples():
    data = xr.DataArray(np.empty((500, 100)), dims=['time', 'x'])
    bs = bootstrap_samples(data, 'time', 1000)
    assert bs.sizes['sample'] == 1000
    repr(bs)                    # this should not fail
