import numpy as np
import xarray as xr

from arch import bootstrap

_null_arr = np.empty(0)

def create_mock_bootstrap(cls, num_items, *bs_params):
    """Creates a mock instance of cls with num_items set without data"""
    args = bs_params + (_null_arr,)         # add empty data
    bs = cls(*args)
    if num_items:
        bs._num_items = num_items  # TODO ugly workaround
    return bs


def bootstrap_samples(xrobj, dim:str, n_samples:int=1000, bootstrap_cls='IIDBootstrap', *bs_params):
    """Create bootstrap samples of the the xarray object along the specified dimension

    Parameters
    ----------
    xrobj : xarray.DataArray or xarray.Dataset
        the object to be bootstrapped along *dim*
    dim : str
        dimension along which to bootstrap xrobj
        .e.g 'time'
    n_samples : int, optional
        how many samples to generate, by default 1000
        will be associated with the new 'sample' dimension
    bootstrap_cls : str or arch.bootstrap.IIDBootstrap subclass
        class to use for the bootstrap procedure, by default IIDBootstrap
        for correlated time series the StationaryBootstrap is a better choice
    *bs_params
        extra params to *bootstrap_cls*, typically the blocksize

    Returns
    -------
    bootstrap_samples : xarray.DataArray or xarray.Dataset
        broadcasted xrobj with an extra 'sample' dimension

    Notes
    -----
    The indices of bootstrap resamples are currently generated
    in a Python *for* cycle and is quite slow.

    xrobj is then simply indexed in a vectorized fashion by the 2D index array.
    """
    n_items = xrobj.sizes[dim]
    if isinstance(bootstrap_cls, str):
        bootstrap_cls = getattr(bootstrap, bootstrap_cls)
    bs = create_mock_bootstrap(bootstrap_cls, n_items, *bs_params)
    indices = [bs.update_indices() for i in range(n_samples)]
    indices = xr.DataArray(np.vstack(indices), dims=['sample', dim])
    res = xrobj.isel(**{dim: indices})
    if dim in res.coords:  # should not be as resampling jumbles the coordinate meaning
        del res.coords[dim] # workaround for old xarray bug, should not be set to 2D index array
    return res

