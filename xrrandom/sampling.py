"""Virtual sample generation"""

import numpy as np

import dask
import dask.array as da
import xarray as xr


"""Empty 1byte array to be strided as required"""
_empty_1d = np.empty(1, dtype='i1')


def _virtual_array(shape:tuple or int):
    """Create a virtual array with an arbitrary shape and just 1 byte allocated

    this uses 0 strides in all axes, so is generally very unsafe!
    It is only useful for carrying shape information.
    """
    if isinstance(shape, int):
        shape = (shape,)
    strides = (0,) * len(shape)
    return np.lib.stride_tricks.as_strided(_empty_1d, shape, strides)


def _ensure_xarray_return_type(ret):
    """Ensures that the return value is a DataArray or Dataset

    if not, wraps it as DataArray(ret)

    if ret is a Variable the dimensions are retained
    otherwise they cannot be guessed
    """
    if not isinstance(ret, (xr.DataArray, xr.Dataset)):
        return xr.DataArray(ret)  # uses existing dimension info if any
    else:
        return ret


def _generate_apply_ufunc(gen_func, args, samples_arr, samples, output_dtype):
    result = xr.apply_ufunc(gen_func, xr.Variable('sample', samples_arr), *args,
                            dask='parallelized', output_dtypes=[output_dtype],
    )
    return _ensure_xarray_return_type(result)


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
        samples_arr = _virtual_array(samples)
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

    samples_arr = da.from_array(_virtual_array(samples), name=SAMPLE_VEC_KEY, chunks=sample_chunksize or samples)
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
    reshaped_virtually_sampled_darray : xarray.DataArray
        the sample dimension has the new shape

    Raises
    ------
    ValueError
        when the sample dimension is chunked,
        because the dask graph could be too complicated to change in that case
    """
    old_data = virtually_sampled_darray.data
    sample_axis = virtually_sampled_darray.get_axis_num('sample')
    if len(old_data.chunks[sample_axis]) != 1:
        raise ValueError('Changing virtual sample count not possible on chunked sample dimension')

    dask_layers = virtually_sampled_darray.data.dask.layers.copy()
    # assign new dict to prevent affecting other dask graphs
    dask_layers[SAMPLE_VEC_KEY] = {
        (SAMPLE_VEC_KEY, 0): _virtual_array(new_sample_count)
    }
    new_shape = list(old_data.shape)
    new_shape[sample_axis] = new_sample_count
    new_chunks = list(old_data.chunks)
    new_chunks[sample_axis] = (new_sample_count,)
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


def sample(distr, samples=None):
    """Sample virtual distribution

    Parameters
    ----------
    distr : xarray.DataArray
        xarray representing the virtual distribution
    samples : int, optional
        number of samples to be generated, defaults to the number of samples specified
        when creating the distribution


    Returns
    -------
    samples : xarray object
        samples from the given distribution
    """
    if not isinstance(distr.data, da.Array):
        raise TypeError('`distribution` must be dask xarray')
    if samples is not None:
        distr = change_virtual_samples(distr, new_sample_count=samples)
    
    return distr.compute()

