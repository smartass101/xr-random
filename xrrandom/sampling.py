"""Virtual sample generation"""

import numpy as np

import dask
import dask.array as da
import xarray as xr



def _generate_apply_ufunc(gen_func, args, samples_arr, samples, output_dtype):
    result = xr.apply_ufunc(gen_func, xr.Variable('sample', samples_arr), *args,
                            dask='parallelized', output_dtypes=[output_dtype],
    )
    return result


def generate_samples(gen_func, args, samples:int=1, output_dtype=np.float):
    """Generate samples using gen_func(samples, *args)


    Parameters
    ----------
    gen_func : callable
        numpy-broadcasting-compatible function to be called as gen_func(samples, *args)
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


def change_virtual_samples(virtually_sampled_darray, new_sample_count:int,
                           new_sample_chunksize:int=None):
    """Change the number of virtual samples

    Parameters
    ----------
    virtually_sampled_darray : xarray.DataArray with dask data
        result of generate_virtual_samples
    new_sample_count : int
        new sample count
    new_sample_chunksize : int, optional
        new sample chunksize, if not provided, the original is used,
        unless the original equals the old sample count, then new_sample_count is used

    Returns
    -------
    reshaped_virtually_sampled_darray : xarray.DataArray
        the sample dimension will have a new size and optionally a new chunksize
    """
    old_data = virtually_sampled_darray.data
    sample_axis = virtually_sampled_darray.get_axis_num('sample')
    new_chunks = list(old_data.chunks)
    if new_sample_chunksize is None:
        if len(old_data.chunks[sample_axis]) == 1:  # just 1 chunk
            new_sample_chunksize = new_sample_count  # do the same but with new sample size
            sample_chunks = (new_sample_chunksize,)
        else:                            # more chunks
            new_sample_chunksize = old_data.chunks[sample_axis][0]  # use the same
            whole_chunks, last_chunk = divmod(new_sample_count, new_sample_chunksize)
            sample_chunks = (new_sample_chunksize,) * whole_chunks
            if last_chunk != 0:
                sample_chunks += (last_chunk,)
    new_chunks[sample_axis] = sample_chunks
    dask_layers = virtually_sampled_darray.data.dask.layers.copy()
    dask_layers[SAMPLE_VEC_KEY] = {
            (SAMPLE_VEC_KEY, i): np.full((chunksize,), chunksize)
        for i, chunksize in enumerate(sample_chunks)
        }
    new_shape = list(old_data.shape)
    new_shape[sample_axis] = new_sample_count
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

