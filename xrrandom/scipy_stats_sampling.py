from itertools import zip_longest
import numpy as np
import xarray as xr

from .scipy_stats_gen import sample_dim_rvs_factory, distribution_kind, get_stats_distribution, distribution_parameters
from .sampling import generate_samples, generate_virtual_samples

_output_dtypes = {
    'continuous': np.float,
    'discrete': np.int,
}


def _parse_scipy_args(param_names, *args, **kwargs):
    # ugly args, kwargs parsing due to inconvenient scipy.stats convention
    args_full = []
    args = list(args)
    missing = []
    for p in param_names:
        if p in kwargs:
            args_full.append(kwargs[p])
        elif len(args) == 0:  # already empty, need defaults
            if p == 'loc':
                args_full.append(0)
            elif p == 'scale':
                args_full.append(1)
            else:
                missing.append(f"'{p}'")
        else:
            args_full.append(args.pop(0))

    if len(missing) > 0:
        raise TypeError(f'missing shape parameter(s): {", ".join(missing)}')

    return args_full


def _rvs_gen_args(stats_distribution, *args, **kwargs):
    stats_distribution = get_stats_distribution(stats_distribution)
    output_dtype = _output_dtypes[distribution_kind(stats_distribution)]
    rvs_gen = sample_dim_rvs_factory(stats_distribution)
    param_names = distribution_parameters(stats_distribution)
    args_full = _parse_scipy_args(param_names, *args, **kwargs)
    return rvs_gen, args_full, output_dtype


def sample_distribution(stats_distribution, samples=1, *args, **kwargs):
    """Sample a distribution from scipy.stats with parameters given as xarray objects


    Parameters
    ----------
    stats_distribution : str or scipy.stats.rv_continuous or scipy.stats.rv_discrete
        name of a scipy.sats distribution or a specific distribution object
    samples : int, optional
        number of samples to draw, defaults to 1
    *args, **kwargs : scalar or  array_like or xarray objects
        positional and keyword arguments to the rvs() method of *stats_distribution*


    Returns
    -------
    samples : xarray object
        samples from the given distribution with argument broadcasted according to dimensions
        and a new dimension 'sample' with size *samples*

    Raises
    ------
    ValueError
        if the *stats_distribution* cannot be found or is not a valid distribution object
    """
    rvs_gen, args_full, output_dtype = _rvs_gen_args(stats_distribution, *args, **kwargs)
    vs = generate_samples(rvs_gen, args_full, samples, output_dtype)
    return vs


def virtually_sample_distribution(stats_distribution, samples=1, *args, sample_chunksize=None, **kwargs):
    """Sample a distribution into dask from scipy.stats with parameters given as xarray objects

    For the virtual sampling idea look at :py:func`xrrandom.sampling.generate_virtual_samples``

    Parameters
    ----------
    stats_distribution : str or scipy.stats.rv_continuous or scipy.stats.rv_discrete
        name of a scipy.sats distribution or a specific distribution object
    samples : int, optional
        number of samples to draw, defaults to 1
    sample_chunksize : ints, optional
        if given, he smaple dimension will have this chunksize
    *args, **kwargs : scalar or  array_like or xarray objects
        positional and keyword arguments to the rvs() method of *stats_distribution*


    Returns
    -------
    samples : xarray object
        samples from the given distribution with argument broadcasted according to dimensions
        and a new dimension 'sample' with size *samples*
        the data will be a dask Array

    Raises
    ------
    ValueError
        if the *stats_distribution* cannot be found or is not a valid distribution object
    """
    rvs_gen, args_full, output_dtype = _rvs_gen_args(stats_distribution, *args, **kwargs)
    vs = generate_virtual_samples(rvs_gen, args_full, samples, output_dtype, sample_chunksize)
    return vs
