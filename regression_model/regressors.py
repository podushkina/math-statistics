import numpy as np

ALL = 1

__REGRESSORS_TYPE_COUNT = 4
__MAX_POLYNOM_POWER = 10
__MAX_COEFFICIENT = 20
REGRESSORS_COUNT = __MAX_COEFFICIENT * __REGRESSORS_TYPE_COUNT


class RegType:
    pass


class FuncsWithCoefficient(RegType):
    def __init__(self, max_coefficient):
        self.max_coefficient = max_coefficient


class Polynomial(FuncsWithCoefficient):
    pass


class Exponential(FuncsWithCoefficient):
    pass


class Sinusoid(FuncsWithCoefficient):
    pass


class CoSinusoid(FuncsWithCoefficient):
    pass
