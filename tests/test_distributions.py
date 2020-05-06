import scipy
import numpy as np
import xarray as xr

import xrrandom
from xrrandom import sample

def test_distribution_exists(stats_distr):
    assert hasattr(xrrandom.distributions, stats_distr['name'])


def test_rvs_pdf(stats_distr, loc=0.5, scale=0.1):
    scipy_distr = stats_distr['distr']
    shape_params = stats_distr['params']

    xr_distr = getattr(xrrandom.distributions, stats_distr['name'])

    if stats_distr['kind'] == 'continuous':
        x = np.linspace(max(scipy_distr.a, -100), min(scipy_distr.b, 100), 1000)
        shape_params = list(shape_params) + [loc, scale]
        method = 'pdf'
    elif stats_distr['kind'] == 'discrete':
        x = np.arange(max(scipy_distr.a, -100), min(scipy_distr.b + 1, 101))
        method = 'pmf'
    else:
        raise ValueError(f'unknown kind {stats_distr["kind"]}')

    xr_shape_params = [xr.DataArray(sp) for sp in shape_params]

    N = 100
    np.random.seed(0)
    scipy_rvs = scipy_distr(*shape_params).rvs(size=N)
    np.random.seed(0)
    xr_rvs = xr_distr(*xr_shape_params)

    assert np.allclose(scipy_rvs, sample(xr_rvs, N), equal_nan=True)
    assert isinstance(xr_rvs, xr.DataArray)
