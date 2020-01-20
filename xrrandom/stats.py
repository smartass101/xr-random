import functools
import inspect
import dask.array as da
import scipy.stats as stats

from .scipy_stats_sampling import sample_distribution, virtually_sample_distribution
from .sampling import change_virtual_samples


def sample(distribution, samples=None):
    """Sample virtual distribution

    Parameters
    ----------
    distribution : xarray.DataArray
        xarray representing the virtual distribution
    samples : int, optional
        number of samples to be generated, defaults to the number of samples specified
        when creating the distribution


    Returns
    -------
    samples : xarray object
        samples from the given distribution
    """
    if not isinstance(distribution.data, da.Array):
        raise TypeError('`distribution` must be dask xarray')
    if samples is not None:
        distribution = change_virtual_samples(distribution, new_sample_count=samples)
    
    return distribution.compute()


class ScipyStatsWrapper:
    """Xarray wrapper for scipy.stats distribution. 
    
    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the standard deviation
    Other shape parameters specific for the distribution are passed to 
    the scipy distribution.

    The size of the output array is determined by broadcasting the shapes of the
    input arrays, scipy.stats ``shape`` parameter is ignored.

    The ``samples`` parameter determines how many random samples will be drawn.

    It contains the same methods as scipy.stats distribution, such as ``rvs`` to draw
    random samples. Direct call creates frozen distribution. 

    Use ``rvs`` with parameter ``virtual=True`` to obtain virtually sampled distribution.

    For the virtual sampling idea look at :py:func`xrrandom.sampling.generate_virtual_samples``
    """

    _default_virtual = False

    def __init__(self, distr):
        self._distr = distr
        self.__doc__ = distr.__doc__.split('\n')[0]

    def rvs(self, *args, samples=1, virtual=None, sample_chunksize=None, **kwargs):
        """Sample the given distribution.
        
        Parameters
        ----------
        args, kwargs: xarray.DataArray
            The shape parameter(s) for the distribution. Should include all the non-optional arguments,
            may include ``loc`` and ``scale``.
        samples: int, optional
            Number of random samples (default: 1)
        virtual: bool, optional
            Return virtually sampled distribution (default: False)
        sample_chinksize: ints, optional
            Chunksize of the sample dimension (default: None). If given, the number of samples cannot be later changed. 
            Used only if `virtual` is True.
        
        Returns
        -------
        rvs: xarray.DataArray
            Random samples from the given distribution. Standard static array 
            if virtual is False,otherwise DataArray wrapping delayed Dask array.
            In such case samples obtained by `virtual_samples.values` will be 
            different on each call. Use `xrrandom.change_virtual_samples` or
            `xrrandom.distributions.sample` to change number of virtual samples.
        """
        virtual = virtual or self._default_virtual

        if virtual:
            return virtually_sample_distribution(self._distr, samples=samples, *args, 
                                             sample_chunksize=sample_chunksize, **kwargs)
        else:
            return sample_distribution(self._distr, samples=samples, *args, **kwargs)
        

    def __call__(self, *args, samples=None, virtual=None, sample_chunksize=None, **kwargs):
        """Freeze the distribution.
        
        Parameters
        ----------
        args, kwargs: xarray.DataArray
            The shape parameter(s) for the distribution. Should include all the non-optional arguments,
            may include ``loc`` and ``scale``.
        samples: int, optional
            Number of random samples. 
        virtual: bool, optional
            Return virtually sampled distribution?
        sample_chinksize: ints, optional
            Chunksize of the sample dimension. If given, the number of samples cannot be later changed. 
            Used only if `virtual` is True.
        
        Returns
        -------
        rvs: FrozenScipyStatsWrapper
            Frozen version of the distribution with shape parameters fixed
        """

        return FrozenScipyStatsWrapper(self, *args, samples=samples, virtual=virtual, sample_chunksize=sample_chunksize, **kwargs)

class FrozenScipyStatsWrapper:
    """Xarray wrapper for frozen scipy.stats distribution.         
    
    It contains the same methods as scipy.stats distribution, such as ``rvs`` to draw
    random samples.

    Use ``rvs`` with parameter ``virtual=True`` to obtain virtually sampled distribution.

    For the virtual sampling idea look at :py:func`xrrandom.sampling.generate_virtual_samples``
    """    
    def __init__(self, distr, *args, samples=None, sample_chunksize=None, virtual=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.distr = distr
        self.samples = samples
        self.sample_chunksize=sample_chunksize
        self.virtual = virtual

    def rvs(self, samples=None, virtual=None, sample_chunksize=None):
        """Sample frozen distribution.
        
        Parameters
        ----------       
        samples: int, optional
            Number of random samples (default: 1)
        virtual: bool, optional
            Return virtually sampled distribution (default: False)
        sample_chinksize: ints, optional
            Chunksize of the sample dimension (default: None). If given, 
            the number of samples cannot be later changed. Used only if 
            `virtual` is True.
        
        Returns
        -------
        rvs: xarray.DataArray
            Random samples from the given distribution. Standard static array 
            if virtual is False, otherwise DataArray wrapping delayed Dask 
            array. In such case samples obtained by `virtual_samples.values` 
            will be different on each call. Use `xrrandom.change_virtual_samples`
            or`xrrandom.distributions.sample` to change number of virtual samples.
        """
        samples = samples or self.samples or 1
        virtual = virtual or self.virtual or self.distr._default_virtual
        sample_chunksize = sample_chunksize or self.sample_chunksize

        return self.distr.rvs(*self.args, samples=samples, virtual=virtual, sample_chunksizes=sample_chunksize, **self.kwargs)

# augment this module by all distributions in the scipy.stats
for name, distr in stats.__dict__.items():
    if isinstance(distr, (stats.rv_continuous, stats.rv_discrete)):
        globals()[name] = ScipyStatsWrapper(distr)
