# xr-random: random data generation parametrized by xarray objects

One of the main uses for this library is to generate Monte Carlo samples from distributions parametrized by xarray objects for the purpose of error propagation in complicated data processing pipelines.

## Example usage


The distributions from `scipy.stats` are used by default

    import numpy as np
    import xarray as xr
    import xrrandom
    loc = xr.DataArray(np.arange(5), dims=['loc'], name='loc')
    scale = xr.DataArray(np.arange(3)/2, dims=['scale'], name='scale')
    # 10 samples with the loc and scale (specified as keyword) automatically broadcasted
    samples = xrrandom.sample_distribution('norm', 10, loc, scale=scale)
    
    
### Virtual samples

In some cases (e.g. interactive usage) it is convenient to
1. define virtual samples with unknown sample count using dask
2. perform various operations on the samples, dask builds the graph
3. compute as many samples as needed (e.g. tested by histograms) from the result

For this use case the `virtually_sample_distribution` function (canonically just 1 virtual sample is used) can be used at the beginning and once as specific number of samples is requested, the `change_virtual_samples` function modifies the number of samples.


    samples = virtually_sample_distribution('norm', 10, loc, scale=scale)
    samples_larger_dask = change_virtual_samples(samples*2, 100)
    samples_larger = samples_larger_dask.compute()

The `sample_chunksize` parameter can be optionally used to possibly parallelize the random sample generation and propagation.


### Bootstrapping

The `bootstrap_samples` function wraps [bootstrapping classes from the `arch` package](https://arch.readthedocs.io/en/latest/bootstrap/bootstrap.html). 
The bootstrapping along a specified dimension lets `xarray` handle any broadcasting of extra dimensions, so it also works for n-dimensional arrays.
A simple IID resampling of a time series can be done like so

    import numpy as np
    import xarray as xr
    from xrrandom import bootstrap_samples

    t = np.linspace(0, 10, 100)
    data = xr.DataArray(np.sin(t), coords=[('time', t)], name='sig')
    resampled = bootstrap_samples(data, 'time', 1000)  # 1000 IIDBootstrap samples by default
    means = resampled.mean(dim='time')  # bootstrap distribution of means
    mean = means.mean(dim='sample')
    mean_unc = means.std(dim='sample')

When the character of the time series permits it, it may be useful to use one of the block-bootstrap classes

    resampled = bootstrap_samples(data, 'time', 1000, 'StationaryBootstrap', 5)  # 5 pts block
