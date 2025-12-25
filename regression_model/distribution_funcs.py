from scipy import stats


def laplace_func(x):
    return stats.norm.cdf(x) - 0.5
