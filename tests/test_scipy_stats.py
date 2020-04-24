import numpy as np
import xarray as xr

import xrrandom


def test_distribution_kind(stats_distr):
    assert getattr(xrrandom.stats, stats_distr['name'])._kind == stats_distr['kind']


def test_rvs_pdf(stats_distr, loc=0.5, scale=0.1):
    scipy_distr = stats_distr['distr']
    shape_params = stats_distr['params']

    xr_distr = getattr(xrrandom.stats, stats_distr['name'])

    if stats_distr['kind'] == 'continuous':
        x = np.linspace(max(scipy_distr.a, -100), min(scipy_distr.b, 100), 1000)
        shape_params = list(shape_params) + [loc, scale]
        method = 'pdf'
    elif stats_distr['kind'] == 'discrete':
        x = np.arange(max(scipy_distr.a, -100), min(scipy_distr.b + 1, 101))
        method = 'pmf'
    else:
        raise ValueError(f'unknown kind {stats_distr["kind"]}')

    shape_params = [xr.DataArray(sp) for sp in shape_params]

    frozen_scipy = scipy_distr(*shape_params)
    scipy_pdf = getattr(frozen_scipy, method)(x)

    assert np.allclose(scipy_pdf, getattr(xr_distr, method)(x, *shape_params), equal_nan=True)
    assert np.allclose(scipy_pdf, getattr(xr_distr(*shape_params), method)(x), equal_nan=True)  # frozen

    N = 100
    np.random.seed(0)
    scipy_rvs = frozen_scipy.rvs(size=N)
    np.random.seed(0)
    xr_rvs = xr_distr.rvs(*shape_params, samples=N)
    assert np.allclose(scipy_rvs, xr_rvs, equal_nan=True)
    assert isinstance(xr_rvs, xr.DataArray)

    np.random.seed(0)
    xr_rvs = xr_distr(*shape_params).rvs(samples=N)
    assert np.allclose(scipy_rvs, xr_rvs, equal_nan=True)
    assert isinstance(xr_rvs, xr.DataArray)
