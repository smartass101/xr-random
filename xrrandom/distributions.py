import scipy.stats as stats

from .scipy_stats_sampling import virtually_sample_distribution
from .scipy_stats_gen import distribution_kind, distribution_parameters
from .stats import _shape_params_to_doc, _wrap_stats_func

class VirtualDistribution:
    """Representation of scipy.stats distribution using virtual samples.
    
    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the standard deviation
    Other shape parameters specific for the distribution are passed to 
    the scipy distribution.

    The size of the output array is determined by broadcasting the shapes of 
    the input arrays, scipy.stats ``shape`` parameter is ignored.

    The ``samples`` parameter determines how many random samples will be drawn,
    can be changed later using ``xrrandom.sample`` or 
    ``xrrandom.change_virtual_samples``.

    For the virtual sampling idea look at 
    :py:func`xrrandom.sampling.generate_virtual_samples``
    """   

    def __init__(self, distr):
        self._distr = distr    
        self.__doc__ = distr.__doc__.split('\n')[0]+'\n\nVirtual distribution.\n\n'

    def _call(self, *args, samples=1, virtual=None, sample_chunksize=None, **kwargs):
        """Create virtually sampled distribution with the given shape.
        
        Parameters
        ----------
        {shape_parameters}
        samples: int, optional
            Number of random samples (default: 1)        
        sample_chinksize: ints, optional
            Chunksize of the sample dimension (default: None). If given, the 
            number of samples cannot be later changed. Used only if `virtual`
            is True.
        
        Returns
        -------
        distr: xarray.DataArray
            DataArray wrapping delayed Dask array. Samples obtained by 
            `distr.values` will be different on each call. Use 
            `xrrandom.change_virtual_samples` or`xrrandom.sample` to 
            change the number of virtual samples.
        """        

        return virtually_sample_distribution(self._distr, samples, *args,
                                             sample_chunksize=sample_chunksize,
                                             **kwargs)


def _register_virtual_rv(name, distr):
    try:
        distribution_kind(distr)
    except ValueError:
        return

    gen_class = type(f'{name}_gen', (VirtualDistribution,), {})
    setattr(gen_class, '__call__', _wrap_stats_func(distr, gen_class._call,
                                                    is_stats_method=False,
                                                    all_pos_or_kw=True))

    globals()[f'_{name}_gen'] = gen_class
    globals()[name] = gen_class(distr)

# augment this module by all distributions in the scipy.stats
for name, distr in vars(stats).items():
    _register_virtual_rv(name, distr)
