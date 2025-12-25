from scipy import stats


def u(level):
    result = stats.norm.ppf(level)
    return result


def t(degree, level):
    result = stats.t.ppf(level, df=degree)
    return result


def x(degree, level):
    result = stats.chi2.ppf(level, df=degree)
    return result


if __name__ == "__main__":
    print(t(3, 0.9))
