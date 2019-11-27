"""Virtual sample generation"""

import numpy as np

import dask
import dask.array as da
import xarray as xr



def _generate_apply_ufunc(gen_func, args, samples_arr, samples, output_dtype):
    result = xr.apply_ufunc(gen_func, xr.Variable('sample', samples_arr), *args,
                            dask='parallelized', output_dtypes=[output_dtype],
                            output_sizes={'sample': samples})
    return result


def generate_samples(gen_func, args, samples:int=1, output_dtype=np.float):
    """Generate samples using gen_func(samples, *args)


    Parameters
    ----------
    gen_func : callable
        numpy-broadcasting-compatible function to be called as gen_func(samples, *args)
        the samples dimension must be returned as last (because of xarray.apply_ufunc core dims)
    args : tuple of numpy.ndarray, xarray objects
        arguments to gen_func to be applied through xarray.apply_ufunc
    samples : int, optional
        number of samples, defaults to 1
    output_dtype : numpy.dtype, optional
        for dask='parallelized' in apply_ufunc, defaults to float

    Returns
    -------
    result : array_like or xarray object
        result of gen_func(samples, *args) with a new 'sample' dimension of size *samples*
    """
    if isinstance(samples, int):
        samples_arr = np.full((samples,), samples)
    result = _generate_apply_ufunc(gen_func, args, samples_arr, samples, output_dtype)
    return result


SAMPLE_VEC_KEY = '__sample-count__'


def generate_virtual_samples(gen_func, args, samples:int=1, output_dtype=np.float, sample_chunksize:int=None):
    """Generate virtual (dask) samples using gen_func(samples, *args)


    Parameters
    ----------
    gen_func : callable
        numpy-broadcasting-compatible function to be called as gen_func(samples, *args)
        the samples dimension must be returned as last (because of xarray.apply_ufunc core dims)
    args : tuple of numpy.ndarray, xarray objects
        arguments to gen_func to be applied through xarray.apply_ufunc
    samples : int or array_like, optional
        number of samples, defaults to 1
        if array_like, will be used directly for the 'sample' dimension broadcasting
    output_dtype : numpy.dtype, optional
        for dask='parallelized' in apply_ufunc, defaults to float
    sample_chunksize : int, optional
        if given, the resulting 'sample' dimension will have this chunksize
        useful when generator function should be parallelized

    Returns
    -------
    result : dask.array.Array or xarray object
        result of gen_func(samples, *args) with a new 'sample' dimension of size *samples*
        the data is a dask Array, so .compute() must be called to get actual samples

    Notes
    -----
        If no sample_chunksize was given, the sample count can be changed later
        (e.g. after performing operations and discovering not enough samples were requested)
        by change_virtual_samples.
        Therefore, it is often convenient to define a virtual sample with sample size 1,
        perform any operations and call compute() after changing the number of samples.
    """
    if sample_chunksize is None:
        samples_arr = da.from_array(np.full((samples,), samples), name=SAMPLE_VEC_KEY)
    else:
        samples_arr = da.from_array(np.full((samples,), sample_chunksize), chunks=sample_chunksize,
                              name=SAMPLE_VEC_KEY)
    result = _generate_apply_ufunc(gen_func, args, samples_arr, samples, output_dtype)
    return result


def change_virtual_samples(virtually_sampled_darray, new_sample_count:int):
    """Change the number of virtual samples

    This works only when no sample chunking was used

    Parameters
    ----------
    virtually_sampled_darray : xarray.DataArray with dask data
        result of generate_virtual_samples with sample_chunksize=None
    new_sample_count : int
        new sample count

    Returns
    -------
    virtually_sampled_darray : xarray.DataArray
        array with a different sample count in the dask graph, but same reported shape
        when .compute() is called, the result will have the new size
    """
    virtually_sampled_darray
    dask_layers = virtually_sampled_darray.data.dask.layers.copy()
    if len(dask_layers[SAMPLE_VEC_KEY]) == 1:
        # assign new dict to prevent affecting other dask graphs
        dask_layers[SAMPLE_VEC_KEY] = {
            (SAMPLE_VEC_KEY, 0): np.full((new_sample_count,), new_sample_count)
        }
    else:
        raise ValueError('Changing virtual sample count not possible on chunked array')
    old_data = virtually_sampled_darray.data
    sample_axis = virtually_sampled_darray.get_axis_num('sample')
    new_shape = list(old_data.shape)
    new_shape[sample_axis] = new_sample_count
    new_chunks = list(old_data.chunks)
    new_chunks[sample_axis] = (new_sample_count,)  # TODO could handle more chunks as well
    new_data = da.Array(dask.highlevelgraph.HighLevelGraph(dask_layers,
                                             old_data.dask.dependencies),
                                             name=old_data.name,
                                             chunks=tuple(new_chunks),
                                             dtype=old_data.dtype,
                                             shape=tuple(new_shape),
    )
    new_darray = xr.DataArray(new_data, coords=virtually_sampled_darray.coords,
                              dims=virtually_sampled_darray.dims,
                              name=virtually_sampled_darray.name,
                              attrs=virtually_sampled_darray.attrs,
    )
    return new_darray

