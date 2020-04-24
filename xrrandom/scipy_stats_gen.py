"""Wrapper of scipy.stats distribution rvs methods"""
import numpy as np
from scipy import stats


def distribution_kind(stats_distribution):
    """Return names of distribution parameters

    Parameters
    ----------
    stats_distribution : scipy.stats.rv_continuous or scipy.stats.rv_discrete
        distribution to inspect

    Returns
    -------
    kind : str
       'continuous' or 'discrete'

    Raises
    ------
    ValueError
        when stats_distribution is not the required type
    """
    if isinstance(stats_distribution, stats.rv_continuous):
        return 'continuous'
    elif isinstance(stats_distribution, stats.rv_discrete):
        return 'discrete'
    else:
        raise ValueError('stats_distribution must be either rv_continuous or rv_discrete')


_general_params = {
    'continuous': ('loc', 'scale'),
    'discrete': ('loc',),
}


def distribution_parameters(stats_distribution):
    """Return names of distribution parameters

    Parameters
    ----------
    stats_distribution : scipy.stats.rv_continuous or scipy.stats.rv_discrete
        distribution to inspect

    Returns
    -------
    parameters : tuple of str
        names of distribution parameters, typically something like
        ('a', 'b', 'loc', 'scale')
        the general parameters loc and scale (if continuous) are last,
        this corresponds to the calling order convention in stats methods

    Raises
    ------
    ValueError
        when stats_distribution is not the required type
    """
    d_kind = distribution_kind(stats_distribution)
    shape_params_str = getattr(stats_distribution, 'shapes', '')
    if shape_params_str is not None and len(shape_params_str) > 0:
        shape_params = tuple(map(str.strip, shape_params_str.split(',')))
    else:
        shape_params = ()
    # scipy.stats ordering convention
    parameters = shape_params + _general_params[d_kind]
    return parameters


def get_stats_distribution(stats_distribution):
    if isinstance(stats_distribution, str):
        try:
            stats_distribution = getattr(stats, stats_distribution)
        except AttributeError:
            raise ValueError('{} not found in scipy.stats'.format(stats_distribution))
    return stats_distribution


def sample_dim_rvs_factory(stats_distribution):
    stats_distribution = get_stats_distribution(stats_distribution)
    d_kind = distribution_kind(stats_distribution)  # check type

    def rvs(sample_vec, *args):
        binfo = np.broadcast(sample_vec, *args)
        result = stats_distribution.rvs(*args, size=binfo.shape)
        return result

    return rvs
