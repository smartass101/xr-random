import numpy as np
import xarray as xr

from arch import bootstrap


def create_mock_bootstrap(cls, num_items, *bs_params):
    args = args + ((),)         # add empty data
    bs = cls(*bs_params)
    if num_items:
        bs._num_items = num_items  # TODO ugly workaround
    return bs


def bootstrap_samples(xrobj, dim:str, n_samples:int, bootstrap_cls='IIDBootstrap', *bs_params):
    n_items = xrobj.sizes[dim]
    if isinstance(bootstrap_cls, str):
        bootstrap_cls = getattr(bootstrap, bootstrap_cls)
    bs = create_mock_bootstrap(bootstrap_cls, n_items, *bs_params)
    indices = [bs.update_indices() for i in range(n_samples)]
    indices = xr.DataArray(np.vstack(indices), dims=['sample', dim])
    return xrobj.isel(**{dim: indices})
