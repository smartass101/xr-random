from scipy import stats
from xrrandom.scipy_stats_gen import get_stats_distribution, distribution_parameters


def test_get_distrib():
    norm_distrib = get_stats_distribution('norm')
    assert norm_distrib is stats.norm

def test_distrib_params():
    params = distribution_parameters(stats.gamma)
    assert params == ('a', 'loc', 'scale')
