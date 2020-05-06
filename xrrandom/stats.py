from inspect import signature, Parameter
import scipy.stats as stats
import numpy as np
import xarray as xr

from .sampling import _ensure_xarray_return_type
from .scipy_stats_sampling import (sample_distribution,
                                   virtually_sample_distribution,
                                   _parse_scipy_args, _output_dtypes)
from .scipy_stats_gen import distribution_kind, distribution_parameters


_excluded_distr = ['frechet_r', # deprecated, inconsistent signature of pdf: (*args, **kwargs) instead of (x, *args, **kwargs)
                   'frechet_l', # deprecated, inconsistent signature of pdf: (*args, **kwargs) instead of (x, *args, **kwargs)
                   ]

_scipy_rv_methods = {
    'continuous': {'pdf', 'logpdf', 'fit', 'fit_loc_scale', 'nnlf'},
    'discrete': {'pmf', 'logpmf'},    
}

_common_scipy_rv_methods = {'cdf', 'logcdf', 'sf', 'logsf', 'ppf', 
                            'moment', 'stats', 'entropy', 'expect', 'median',
                            'mean', 'std', 'var', 'interval'}

def _update_signature(sig, shape_params, all_pos_or_kw=True, remove_kwargs=False, add_self=True):    
    params = []
    if 'self' not in sig.parameters and add_self:
        params += [Parameter('self', Parameter.POSITIONAL_OR_KEYWORD)] 

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


def _shape_params_to_doc(shape_parameters, doc):
    shape_doc = ''
    for sp in shape_parameters:
        sp_doc = f'{sp}: array_like'
        if sp == 'scale':
            sp_doc += ', optional (default=1)'
        elif sp == 'loc':
            sp_doc += ', optional (default=0)'
        shape_doc += sp_doc+'\n        '
    shape_doc = shape_doc[:-9]

    return doc.format(shape_parameters = shape_doc)


def _update_signature_and_doc(func, ref_func, shape_parameters,
                              all_pos_or_kw = True, remove_kwargs = False):
    """Modify (inplace) signature and docstring of func to mimic ref_func"""
    sig = signature(ref_func)
    func.__signature__ = _update_signature(sig, shape_parameters,
                                           all_pos_or_kw=all_pos_or_kw,
                                           remove_kwargs=remove_kwargs)
                                           
    if hasattr(ref_func, '__doc__') and ref_func.__doc__ is not None:                
        func.__doc__ = _shape_params_to_doc(shape_parameters, ref_func.__doc__)           


def _wrap_stats_func(stats_distribution, func, is_stats_method,
                     all_pos_or_kw = True, remove_kwargs = False, ):
    """Return func from scipy distribution with updated signature describing
    shape parameters of the distribution"""
            
    sig = signature(func)
    shape_parameters = list(distribution_parameters(stats_distribution))
    pos_parameters = [k for k, v in sig.parameters.items() if v.kind in \
                      [Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD]]

    # TODO: better way how to distinguish rvs and __call__ from scipy methods
    if is_stats_method:
        output_dtype = _output_dtypes[distribution_kind(stats_distribution)]

        def wrapped_method(self, *args, **kwargs):
            args = _parse_scipy_args(pos_parameters + shape_parameters, *args, **kwargs)
            ret =  xr.apply_ufunc(func, *args, output_dtypes=[output_dtype])
            return _ensure_xarray_return_type(ret)

    else:

        def wrapped_method(self, *args, **kwargs):
            return func(self, *args, **kwargs)

    _update_signature_and_doc(wrapped_method, func,
                              shape_parameters,
                              all_pos_or_kw=all_pos_or_kw,
                              remove_kwargs=remove_kwargs)

    return wrapped_method


class ScipyDistributionBase:
    """Xarray wrapper for scipy.stats distribution. 
    
    The location (``loc``) keyword specifies the mean.
    The scale (``scale``) keyword specifies the standard deviation
    Other shape parameters specific for the distribution are passed to 
    the scipy distribution.

    The size of the output array is determined by broadcasting the shapes of 
    the input arrays, scipy.stats ``shape`` parameter is ignored.

    The ``samples`` parameter in ``rvs`` method determines how many random
    samples will be drawn.

    It contains the same methods as scipy.stats distribution, such as ``rvs``    
    to draw random samples or ``pdf`` to compute probability density function. 
    Direct call creates frozen distribution. 

    Use ``rvs`` with parameter ``virtual=True`` to obtain virtually sampled 
    distribution.

    For the virtual sampling idea look at :py:func`xrrandom.sampling.generate_virtual_samples``
    """

    _default_virtual = False    

    def __init__(self, distr):
        self._distr = distr

        # copy docstring
        doc_rows = distr.__doc__.split('\n')
        self.__doc__ = doc_rows[0]+'\n\nXarray wrapper for scipy.stats class.\n\n'
        try:
            first_row = doc_rows.index('    Methods')
        except ValueError:
            first_row = 2
        try:
            last_row =   doc_rows.index('    Examples')
        except ValueError:
            last_row = None
        self.__doc__ += '\n'.join(row[4:] for row in doc_rows[first_row:last_row])

        self._kind = distribution_kind(distr)
        if self._kind == 'continuous':
            self._frozen_class = FrozenScipyContinuous
        elif self._kind == 'discrete':
            self._frozen_class = FrozenScipyDiscrete
        else:
            raise ValueError(f'unknown distribution kind {self._kind}')

    @property
    def shape_parameters(self):
        """Shape parameters of the distribution"""
        return distribution_parameters(self._distr)

    def _rvs(self, *args, samples=1, virtual=None, sample_chunksize=None, **kwargs):
        """Sample the given distribution.
        
        Parameters
        ----------
        {shape_parameters}
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
            if virtual is False,otherwise DataArray wrapping delayed Dask 
            array. In such case samples obtained by `virtual_samples.values` 
            will be different on each call. Use 
            `xrrandom.change_virtual_samples` or 
            `xrrandom.distributions.sample`
            to change number of virtual samples.
        """
        virtual = virtual or self._default_virtual        

        if virtual:
            return virtually_sample_distribution(self._distr, samples, *args, 
                                             sample_chunksize=sample_chunksize, **kwargs)
        else:
            return sample_distribution(self._distr, samples, *args, **kwargs)

    def _call(self, *args, samples=None, virtual=None, sample_chunksize=None, **kwargs):
        """Freeze the distribution.
        
        Parameters
        ----------
        {shape_parameters}
        samples: int, optional
            Number of random samples (default: 1). 
        virtual: bool, optional
            Return virtually sampled distribution? (default: False)
        sample_chinksize: ints, optional
            Chunksize of the sample dimension. If given, the number of samples
            cannot be later changed. Used only if `virtual` is True.
        
        Returns
        -------
        rvs: FrozenScipyContinuous or FrozenScipyDiscrete
            Frozen version of the distribution with shape parameters fixed
        """

        doc = self.__doc__.split('\n')[0] 
        return self._frozen_class(self, *args, samples=samples, virtual=virtual, 
                                  sample_chunksize=sample_chunksize, doc=doc,
                                  **kwargs)


class FrozenScipyBase:
    """Xarray wrapper for frozen scipy.stats distribution.         
    
    It contains the same methods as scipy.stats distribution, such as ``rvs``
    to draw random samples.

    Use ``rvs`` with parameter ``virtual=True`` to obtain virtually sampled
    distribution.

    For the virtual sampling idea look at 
    :py:func``xrrandom.sampling.generate_virtual_samples``
    """    
    def __init__(self, distr, *args, samples=None, sample_chunksize=None, 
                 virtual=None, doc=None, **kwargs):

        self._args = args
        self._kwargs = kwargs
        self._distr = distr
        shape_params = self._distr.shape_parameters
        # _parse_scipy_args raises TypeError if some shape parameters are missing
        values = _parse_scipy_args(shape_params, *self._args, **self._kwargs)
        self._shape_parameters = dict((k, v) for k, v in zip(shape_params, values))

        self.samples = samples
        self.sample_chunksize=sample_chunksize
        self.virtual = virtual

        if doc is not None:
            self.__doc__ = doc

    @property
    def shape_parameters(self):
        return self._shape_parameters

    def cdf(self, x):
        return self._distr.cdf(x, *self._args, **self._kwargs)

    def logcdf(self, x):
        return self._distr.logcdf(x, *self._args, **self._kwargs)

    def sf(self, x):
        return self._distr.sf(x, *self._args, **self._kwargs)

    def logsf(self, x):
        return self._distr.logsf(x, *self._args, **self._kwargs)

    def ppf(self, q):
        return self._distr.ppf(q, *self._args, **self._kwargs)

    def moment(self, n):
        return self._distr.moment(n, *self._args, **self._kwargs)

    def stats(self, moments='mv'):
        return self._distr.stats(*self._args, moments=moments, **self._kwargs)

    def entropy(self):
        return self._distr.entropy(*self._args, **self._kwargs)

    def expect(self, func=None, lb=None, ub=None, conditional=False):
        return self._distr.expect(*self._args, func=func, lb=lb, ub=ub, conditional=conditional,
                                  **self._kwargs)
                                
    def median(self):
        return self._distr.median(*self._args, **self._kwargs)

    def mean(self):
        return self._distr.mean(*self._args, **self._kwargs)

    def std(self):
        return self._distr.std(*self._args, **self._kwargs)

    def var(self):
        return self._distr.var(*self._args, **self._kwargs)

    def interval(self, alpha):
        return self._distr.interval(alpha, *self._args, **self._kwargs)

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
            will be different on each call. Use 
            `xrrandom.change_virtual_samples` or 
            `xrrandom.distributions.sample` 
            to change number of virtual samples.
        """
        samples = samples or self.samples or 1
        virtual = virtual or self.virtual or self._distr._default_virtual
        sample_chunksize = sample_chunksize or self.sample_chunksize

        return self._distr.rvs(*self._args, samples=samples, virtual=virtual, sample_chunksizes=sample_chunksize, **self._kwargs)

    def __repr__(self):
        func_args = ', '.join(f'{k}={v}' for k, v in (list(self.shape_parameters.items())+
                                                           [('samples', self.samples),
                                                            ('sample_chunksize', self.sample_chunksize),
                                                            ('virtual', self.virtual)]))
        distr_name = self._distr.__class__.__name__
        if distr_name.endswith('_gen'):
            distr_name = distr_name[:-4]
        return f'{distr_name}({func_args})'


class FrozenScipyContinuous(FrozenScipyBase):    
    def pdf(self, x):
        return self._distr.pdf(x, *self._args, **self._kwargs)
    
    def logpdf(self, x):
        return self._distr.logpdf(x, *self._args, **self._kwargs)


class FrozenScipyDiscrete(FrozenScipyBase):    
    def pmf(self, k):
        return self._distr.pmf(k, *self._args, **self._kwargs)

    def logpmf(self, k):
        return self._distr.logpmf(k, *self._args, **self._kwargs)


def _register_rv(name, distr):
    """Create new rv class and add it and its parent class to the module's scope"""
    try:
        kind = distribution_kind(distr)
    except ValueError:
        return

    gen_class = type(f'{name}_gen', (ScipyDistributionBase,), {})

    # bind methods methods and update their signature
    for method_name in _common_scipy_rv_methods | _scipy_rv_methods.get(kind, set()):
        orig_method = getattr(distr, method_name)
        new_method = _wrap_stats_func(distr, orig_method, is_stats_method=True,
                                      all_pos_or_kw=True)
        setattr(gen_class, method_name, new_method)

    setattr(gen_class, 'rvs', _wrap_stats_func(distr, gen_class._rvs,
                                               is_stats_method=False,
                                               all_pos_or_kw=True))
    setattr(gen_class, '__call__', _wrap_stats_func(distr, gen_class._call,
                                                    is_stats_method=False,
                                                    all_pos_or_kw=True))                                                            

    globals()[f'_{name}_gen'] = gen_class
    globals()[name] = gen_class(distr)


# augment this module by all distributions in scipy.stats
for name, distr in vars(stats).items():
    if name in _excluded_distr:
        continue
    _register_rv(name, distr)
