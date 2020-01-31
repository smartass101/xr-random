from types import MethodType
from functools import partial
from inspect import signature, Parameter
import dask.array as da
import scipy.stats as stats
import warnings

from .scipy_stats_sampling import sample_distribution, virtually_sample_distribution
from .scipy_stats_gen import distribution_kind, distribution_parameters


_scipy_rv_methods = {
    'continuous': {'pdf', 'logpdf', 'fit', 'fit_loc_scale', 'nnlf'},
    'discrete': {'pmf', 'logpmf'},    
}

_common_scipy_rv_methods = {'cdf', 'logcdf', 'sf', 'logsf', 'ppf', 
                            'moment', 'stats', 'entropy', 'expect', 'median',
                            'mean', 'std', 'var', 'interval'}

def _update_signature(sig, shape_params, all_pos_or_kw=True, remove_kwargs=False):
    params = [Parameter('self', Parameter.POSITIONAL_ONLY)]
    for param in sig.parameters.values():
        if param.kind == Parameter.VAR_POSITIONAL:
            for shape_par in shape_params:
                if shape_par == 'loc':
                    params.append(Parameter('loc', Parameter.POSITIONAL_OR_KEYWORD, default=0))
                elif shape_par == 'scale':
                    params.append(Parameter('scale', Parameter.POSITIONAL_OR_KEYWORD, default=1))
                else:
                    params.append(Parameter(shape_par, Parameter.POSITIONAL_OR_KEYWORD))
        else:
            if all_pos_or_kw and param.kind == Parameter.KEYWORD_ONLY:
                param = param.replace(kind = Parameter.POSITIONAL_OR_KEYWORD)
            if not remove_kwargs or param.kind != Parameter.VAR_KEYWORD:
                params.append(param)

            try:
                shape_params.remove(param.name)
            except ValueError:
                pass
                    
    return sig.replace(parameters=params)

def _wrap_stats_method(stats_distribution, method, all_pos_or_kw = True, remove_kwargs = False,
                       shape_parameters = None):
    """Return method from scipy distribution with updated signature describing
    shape parameters of the distribution"""
        
    def call_scipy_method(self, *args, **kwargs):
        return method(*args, **kwargs)                
    
    if shape_parameters is None:
        shape_parameters = list(distribution_parameters(stats_distribution))

    call_scipy_method.__signature__ = _update_signature(signature(method), 
                                                        shape_parameters,
                                                        all_pos_or_kw=all_pos_or_kw,
                                                        remove_kwargs=remove_kwargs)

    return call_scipy_method
     

def _bind_frozen_method(dest, source, method, *args, **kwargs):
    return partial(getattr(source, method).__get__(dest), *args, **kwargs)


class ScipyDistribution:
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

        self._kind = distribution_kind(distr)
        if self._kind == 'continuous':
            self._frozen_class = FrozenScipyContinuous
        elif self._kind == 'discrete': 
            self._frozen_class = FrozenScipyDiscrete
        else:
            raise ValueError(f'unknown distribution kind {self._kind}')

        # bind methods like pdf, cdf, ...        
        for method_name in _common_scipy_rv_methods | _scipy_rv_methods[self._kind]:
            orig_method = getattr(distr, method_name)
            new_method = MethodType(_wrap_stats_method(distr,orig_method), self)
            setattr(self, method_name, new_method)
    
        # update signature of rvs method
        self.rvs = MethodType(_wrap_stats_method(distr, self._rvs, all_pos_or_kw=True), self)


    def _rvs(self, *args, samples=1, virtual=None, sample_chunksize=None, **kwargs):
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
            return virtually_sample_distribution(self._distr, samples, *args, 
                                             sample_chunksize=sample_chunksize, **kwargs)
        else:
            return sample_distribution(self._distr, samples, *args, **kwargs)
        

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
        rvs: FrozenScipyContinuous or FrozenScipyDiscrete
            Frozen version of the distribution with shape parameters fixed
        """
        
        return self._frozen_class(self, *args, samples=samples, virtual=virtual, sample_chunksize=sample_chunksize, **kwargs)

class FrozenScipyBase:
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

    def cdf(self, x):
        return self.distr.cdf(x, *self.args, **self.kwargs)

    def logcdf(self, x):
        return self.distr.logcdf(x, *self.args, **self.kwargs)

    def sf(self, x):
        return self.distr.sf(x, *self.args, **self.kwargs)

    def logsf(self, x):
        return self.distr.logsf(x, *self.args, **self.kwargs)

    def ppf(self, q):
        return self.distr.ppf(q, *self.args, **self.kwargs)

    def moment(self, n):
        return self.distr.moment(n, *self.args, **self.kwargs)

    def stats(self, moments='mv'):
        return self.distr.stats(*self.args, moments=moments, **self.kwargs)

    def entropy(self):
        return self.distr.entropy(*self.args, **self.kwargs)

    def expect(self, func=None, lb=None, ub=None, conditional=False):
        return self.distr.expect(*self.args, func=func, lb=lb, ub=ub, conditional=conditional,
                                 **self.kwargs)
                                
    def median(self):
        return self.distr.median(*self.args, **self.kwargs)

    def mean(self):
        return self.distr.mean(*self.args, **self.kwargs)

    def std(self):
        return self.distr.std(*self.args, **self.kwargs)

    def var(self):
        return self.distr.var(*self.args, **self.kwargs)

    def interval(self, alpha):
        return self.distr.interval(alpha, *self.args, **self.kwargs)

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

class FrozenScipyContinuous(FrozenScipyBase):
    def pdf(self, x):
        return self.distr.pdf(x, *self.args, **self.kwargs)
    
    def logpdf(self, x):
        return self.distr.logpdf(x, *self.args, **self.kwargs)


class FrozenScipyDiscrete(FrozenScipyBase):
    def pmf(self, k):
        return self.distr.pmf(x, *self.args, **self.kwargs)

    def pmf(self, k):
        return self.distr.logpmf(x, *self.args, **self.kwargs)


# augment this module by all distributions in the scipy.stats
for name, distr in stats.__dict__.items():
    if isinstance(distr, (stats.rv_continuous, stats.rv_discrete)):
        globals()[name] = ScipyDistribution(distr)
