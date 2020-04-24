"""Common fixtures"""
import pytest
import scipy
from scipy.stats._distr_params import distcont, distdiscrete

from xrrandom.stats import _excluded_distr
from xrrandom.scipy_stats_gen import distribution_kind

distcont = dict(distcont)
distdiscrete = dict(distdiscrete)

scipy_distributions = []
for name, distr in vars(scipy.stats).items():
    if name in _excluded_distr:
        continue
    try:
        kind = distribution_kind(distr)
    except ValueError:
        continue
    scipy_distributions.append(
        {'name': name, 'distr': distr, 'kind': kind,
         'params': distcont.get(name) if kind == 'continuous' else distdiscrete.get(name)})

@pytest.fixture(scope='session', params=scipy_distributions)
def stats_distr(request):
    return request.param
