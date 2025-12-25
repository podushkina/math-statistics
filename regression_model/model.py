import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
from threading import Thread

from .regressors import *
from .quantile import x as chi, t, u
from .distribution_funcs import laplace_func


class RegressionModel:
    required_type = np.float64

    def __init__(self, x, y, reg_types: list[Polynomial, Exponential, Sinusoid, CoSinusoid] = ALL, bins=6):
        if len(x) != len(y):
            raise ValueError("!!! len(x) != len(y) !!!")
        self.x: np.array = np.array(x)
        self.Y: np.array = np.array(y)
        if reg_types == ALL:
            self.__reg_types = [Polynomial(__MAX_POLYNOM_POWER), Exponential(__MAX_COEFFICIENT),
                                Sinusoid(__MAX_COEFFICIENT), CoSinusoid(__MAX_COEFFICIENT)]
            self.__reg_types = [Polynomial(__MAX_POLYNOM_POWER),
                                Sinusoid(__MAX_COEFFICIENT), CoSinusoid(__MAX_COEFFICIENT)]
        else:
            self.__reg_types = list(reg_types)
        self.reg_count = sum([func.max_coefficient + 1 if isinstance(func, Polynomial) else func.max_coefficient
                              for func in self.__reg_types])
        # print(f"Количество регрессоров: {self.__reg_count}")
        self.X = np.zeros((self.reg_count, len(self.x)), dtype=self.required_type)
        self.n = len(self.x)
        self.theta = None
        self.y_func = None
        self.y_values = None
        self.alpha_k = None
        self.X_inv = None
        self.sigma_sqr = None
        self.norm_err_sqr = None
        self.x_linspace = np.linspace(min(self.x), max(self.x), 1000)
        self.bins = bins

        self.X_T_X = None
        self.task2_1 = None
        self.task2_2 = None
        self.task4_1 = None
        self.task4_2 = None
        self.task5_1 = None
        self.task5_2 = None
        self.task8_chi_min = None
        self.task8_chi_max = None
        self.task8_statistic = None
        self.alpha_x_func = None
        self.alpha_x = None
        self.alpha_x_view = None

        self.func_lower_alpha1 = None
        self.func_upper_alpha1 = None
        self.func_lower_alpha2 = None
        self.func_upper_alpha2 = None
        self.err_vector = None

        self.coef_alpha_x = np.zeros(self.reg_count * self.reg_count, dtype=np.float64)

    def calculate_only_theta(self):
        index = 0
        for regressor in self.__reg_types:
            if isinstance(regressor, Polynomial):
                self.X[index] = np.ones(len(self.x), dtype=self.required_type)
                index += 1
                for i in range(1, regressor.max_coefficient + 1):
                    self.X[index] = np.pow(self.x, i)
                    index += 1
            elif isinstance(regressor, Exponential):
                for i in range(1, regressor.max_coefficient + 1):
                    self.X[index] = np.exp(self.x * i)
                    index += 1
            elif isinstance(regressor, Sinusoid):
                for i in range(1, regressor.max_coefficient + 1):
                    self.X[index] = np.sin(self.x * i)
                    index += 1
            elif isinstance(regressor, CoSinusoid):
                for i in range(1, regressor.max_coefficient + 1):
                    self.X[index] = np.cos(self.x * i)
                    index += 1

        X_T = self.X.transpose()
        # print(f"Матрица X:\n{X_T}\n")
        # print(f"Матрица X^T:\n{self.X}\n")

        result = np.matmul(self.X, X_T)
        self.X_T_X = result
        # print(f"Матрица (X^T * X):\n{result}\n")

        result = linalg.inv(result)
        self.X_inv = result
        # print(f"Матрица (X^T * X)^-1:\n{result}\n")
        count = len(result[0])
        vector_alpha = np.zeros(count, dtype=np.float64)
        for i in range(count):
            vector_alpha[i] = result[i][i]
        self.alpha_k = vector_alpha

        result = np.matmul(result, self.X)
        # print(f"Матрица ((X^T * X)^-1) * X^T:\n{result}\n")

        # print(f"Значения alpha_k:\n{self.alpha_k}\n\n\n")

        # print(f"КУРСОВАЯ РАБОТА\n")

        self.theta = np.matmul(result, self.Y)
        # print(f"1) МНК оценка коэффициентов theta:\n{self.theta}\n")

    def calculate(self):
        self.calculate_only_theta()

        X_T = self.X.transpose()

        alpha1 = 1 - 0.95
        alpha2 = 1 - 0.99

        matrix = np.matmul(X_T, self.theta)
        matrix = self.Y - matrix
        matrix_t = matrix.transpose()
        result = np.matmul(matrix_t, matrix)
        self.norm_err_sqr = result

        task2_1 = list()
        task2_2 = list()

        for k in range(self.reg_count):
            task2_1.append([self.theta[k] - t(self.n - self.reg_count,
                                              1 - alpha1 / 2) * (self.norm_err_sqr ** 0.5) * (self.alpha_k[k] ** 0.5) * (1 / np.sqrt(self.n - self.reg_count)),
                            self.theta[k] + t(self.n - self.reg_count,
                                              1 - alpha1 / 2) * (self.norm_err_sqr ** 0.5) * (self.alpha_k[k] ** 0.5) * (1 / np.sqrt(self.n - self.reg_count))
                            ])
            task2_2.append([self.theta[k] - t(self.n - self.reg_count,
                                              1 - alpha2 / 2) * (self.norm_err_sqr ** 0.5) * (self.alpha_k[k] ** 0.5) * (1 / np.sqrt(self.n - self.reg_count)),
                            self.theta[k] + t(self.n - self.reg_count,
                                              1 - alpha2 / 2) * (self.norm_err_sqr ** 0.5) * (self.alpha_k[k] ** 0.5) * (1 / np.sqrt(self.n - self.reg_count))
                            ])
        task2_1 = np.array(task2_1)
        task2_2 = np.array(task2_2)
        # print(f"2) Центральные доверительные интервалы для theta уровней надёжности:")
        # print(f" а) 0.95:")
        # print(f"{task2_1}\n")
        # print(f" б) 0.99:")
        # print(f"{task2_2}\n")
        self.task2_1 = task2_1
        self.task2_2 = task2_2

        self.sigma_sqr = (1 / len(self.x)) * result
        # print(f"3) МП оценка дисперсии случайной ошибки:\n{self.sigma_sqr}\n")

        task4_1 = list()
        task4_2 = list()
        task4_1.append([self.norm_err_sqr / chi(self.n - self.reg_count, 1 - alpha1 / 2),
                        self.norm_err_sqr / chi(self.n - self.reg_count, alpha1 / 2)])
        task4_2.append([self.norm_err_sqr / chi(self.n - self.reg_count, 1 - alpha2 / 2),
                        self.norm_err_sqr / chi(self.n - self.reg_count, alpha2 / 2)])
        task4_1 = np.array(task4_1)
        task4_2 = np.array(task4_2)
        # print(f"4) Центральные доверительные интервалы для дисперсии случайной ошибки уровней надёжности:")
        # print(f" а) 0.95:")
        # print(f"{task4_1}\n")
        # print(f" б) 0.99:")
        # print(f"{task4_2}\n")
        self.task4_1 = task4_1
        self.task4_2 = task4_2

        def func(x_):
            _result = 0
            _reg_index = 0
            for _regressor in self.__reg_types:
                if isinstance(_regressor, Polynomial):
                    _result += self.theta[_reg_index]
                    _reg_index += 1
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _result += self.theta[_reg_index] * np.pow(x_, _i)
                        _reg_index += 1
                elif isinstance(_regressor, Exponential):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _result += self.theta[_reg_index] * np.exp(x_ * _i)
                        _reg_index += 1
                elif isinstance(_regressor, Sinusoid):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _result += self.theta[_reg_index] * np.sin(x_ * _i)
                        _reg_index += 1
                elif isinstance(_regressor, CoSinusoid):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _result += self.theta[_reg_index] * np.cos(x_ * _i)
                        _reg_index += 1
            return _result

        self.y_func = func

        def get_string_alpha_x(accuracy=3, epsilon=0.001):
            _count = self.reg_count * 2 - 1
            _result = str()
            _regressors = list()
            _funcs = dict()
            _reg_index = 0
            for _regressor in self.__reg_types:
                if isinstance(_regressor, Polynomial):
                    _regressors.append("1")
                    _reg_index += 1
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _regressors.append(f"x**{_i}")
                        _reg_index += 1
                elif isinstance(_regressor, Exponential):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _regressors.append(f"e**({_i}x)")
                        _reg_index += 1
                elif isinstance(_regressor, Sinusoid):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _regressors.append(f"sin({_i}x)")
                        _reg_index += 1
                elif isinstance(_regressor, CoSinusoid):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _regressors.append(f"cos({_i}x)")
                        _reg_index += 1

            # print(f"LEN: {len(_regressors)}; REGRESSORS: {_regressors}")
            for reg1 in _regressors:
                for reg2 in _regressors:
                    if reg1 == "1":
                        _funcs[reg2] = 1
                    elif ("**" in reg1) and ("**" in reg2):
                        power1 = int(reg1.split("**")[-1])
                        power2 = int(reg2.split("**")[-1])
                        _funcs[f"x**{str(power1 + power2)}"] = 1
                    elif reg2 == "1":
                        pass
                    elif "**" in reg1:
                        _funcs[f"{reg2}*{reg1}"] = 1
                    elif "**" in reg2:
                        _funcs[f"{reg1}*{reg2}"] = 1
                    elif reg1 == reg2:
                        _funcs[f"({reg1}**2)"] = 1
                    else:
                        reg1, reg2 = min(reg1, reg2), max(reg1, reg2)
                        _funcs[f"({reg1} * {reg2})"] = 1
            # print(f"LEN: {len(_funcs)}; FUNCS: {_funcs}")

            _res = 0
            _index = 0
            for _i in range(self.reg_count):
                current_i = _i
                _res = 0
                for _j in range(_i + 1):
                    # print(f"X[{current_i - _j}][{_j}]")
                    _res += self.X_inv[current_i - _j][_j]
                self.coef_alpha_x[_index] = _res
                _index += 1

            """
                        for _i in range(self.reg_count - 1, 0, -1):
                current_i = _i
                _res = 0
                for _j in range(self.reg_count - 1, _i - 1, -1):
                    print(f"X[{current_i - _j + self.reg_count - 1}][{_j}]")
                    _res += self.X_inv[current_i - _j + self.reg_count - 1][_j]
                self.coef_alpha_x[_index] = _res
                _index += 1
                print()
            """
            for _i in range(1, self.reg_count):
                current_i = _i
                _res = 0
                for _j in range(_i, self.reg_count):
                    # print(f"X[{current_i - _j + self.reg_count - 1}][{_j}]")
                    _res += self.X_inv[current_i - _j + self.reg_count - 1][_j]
                self.coef_alpha_x[_index] = _res
                _index += 1

            for _index, _regressor in enumerate(_funcs):
                plus = _regressor
                if _regressor == "1":
                    plus = ""
                if -epsilon < self.coef_alpha_x[_index] < epsilon:
                    continue
                _result += f"({round(self.coef_alpha_x[_index], accuracy)}){plus} + "
            return _result[:-3]

        def alpha_x(_x):
            _result = None
            _reg_index = 0
            _vec = np.zeros(self.reg_count)
            for _regressor in self.__reg_types:
                if isinstance(_regressor, Polynomial):
                    _vec[_reg_index] = 1
                    _reg_index += 1
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _vec[_reg_index] = np.pow(_x, _i)
                        _reg_index += 1
                elif isinstance(_regressor, Exponential):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _vec[_reg_index] = np.exp(_x * _i)
                        _reg_index += 1
                elif isinstance(_regressor, Sinusoid):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _vec[_reg_index] = np.sin(_x * _i)
                        _reg_index += 1
                elif isinstance(_regressor, CoSinusoid):
                    for _i in range(1, _regressor.max_coefficient + 1):
                        _vec[_reg_index] = np.cos(_x * _i)
                        _reg_index += 1
            _result = np.matmul(_vec, self.X_inv)
            _vec = _vec.transpose()
            _result = np.matmul(_result, _vec)
            return _result

        self.alpha_x_func = alpha_x

        # print(f"5) доверительные интервалы для полезного сигнала:")
        alpha1 = 1 - 0.95
        alpha2 = 1 - 0.99
        func_view = self.func_view()

        x = self.x_linspace
        alpha_x = np.zeros_like(x)
        for i, _x in enumerate(x):
            alpha_x[i] = self.alpha_x_func(_x)

        self.alpha_x = alpha_x

        """
        # РАБОТАЕТ ТОЛЬКО ДЛЯ ПОЛИНОМОВ 2 СТЕПЕНИ !!!!!!!!!!!!!!!!!!!!!!!
        model1 = RegressionModel(x, alpha_x, [Polynomial(4)])
        model1.calculate_only_theta()
        self.alpha_x_view = model1.func_view(3)

        alpha_x_view = self.alpha_x_view
        """

        alpha_x_view = get_string_alpha_x()
        self.alpha_x_view = alpha_x_view
        # print(f"КВАНТИЛЬ1 {t(self.n - self.reg_count, 1 - alpha1 / 2)}")
        # print(f"КВАНТИЛЬ2 {t(self.n - self.reg_count, 1 - alpha2 / 2)}")
        coef1 = t(self.n - self.reg_count, 1 - alpha1 / 2) * (self.norm_err_sqr ** 0.5) * (
                    1 / np.sqrt(self.n - self.reg_count))
        coef2 = t(self.n - self.reg_count, 1 - alpha2 / 2) * (self.norm_err_sqr ** 0.5) * (
                    1 / np.sqrt(self.n - self.reg_count))
        self.task5_1 = f"[[{func_view} - {coef1} * sqrt({alpha_x_view});\n  {func_view} + {coef1} * sqrt({alpha_x_view})]]"
        self.task5_2 = f"[[{func_view} - {coef2} * sqrt({alpha_x_view});\n  {func_view} + {coef2} * sqrt({alpha_x_view})]]"
        # print(f"alpha(x) = {alpha_x_view}")
        # print(f" a) 0.95:")
        # print(f"[[{func_view} - {coef1} * sqrt({alpha_x_view});\n  {func_view} + {coef1} * sqrt({alpha_x_view})]]")
        # print()
        # print(f" б) 0.99:")
        # print(f"[[{func_view} - {coef2} * sqrt({alpha_x_view});\n  {func_view} + {coef2} * sqrt({alpha_x_view})]]")
        # print()
        self.func_lower_alpha1 = self.y_func(x) - t(self.n - self.reg_count, 1 - alpha1 / 2) * (self.norm_err_sqr ** 0.5) * (self.alpha_x ** 0.5) * (1 / np.sqrt(self.n - self.reg_count))
        self.func_upper_alpha1 = self.y_func(x) + t(self.n - self.reg_count, 1 - alpha1 / 2) * (self.norm_err_sqr ** 0.5) * (self.alpha_x ** 0.5) * (1 / np.sqrt(self.n - self.reg_count))
        self.func_lower_alpha2 = self.y_func(x) - t(self.n - self.reg_count, 1 - alpha2 / 2) * (self.norm_err_sqr ** 0.5) * (self.alpha_x ** 0.5) * (1 / np.sqrt(self.n - self.reg_count))
        self.func_upper_alpha2 = self.y_func(x) + t(self.n - self.reg_count, 1 - alpha2 / 2) * (self.norm_err_sqr ** 0.5) * (self.alpha_x ** 0.5) * (1 / np.sqrt(self.n - self.reg_count))

        self.y_values = self.y_func(x)
        self.err_vector = self.Y - self.y_func(self.x)
        self.err_vector.sort()

        # self.bins = 6
        n_i = np.zeros(self.bins, dtype=np.float64)
        p_i = np.zeros(self.bins, dtype=np.float64)
        n = len(self.err_vector)

        minimum = self.err_vector[0]
        maximum = self.err_vector[-1] + 0.01
        bins = self.bins
        step = (maximum - minimum) / bins
        index = 0
        for i in np.arange(minimum, maximum, step):
            for err_value in self.err_vector:
                if i <= err_value < i + step:
                    n_i[index] += 1
                    p_i[index] = laplace_func((i + step) / np.sqrt(self.sigma_sqr)) - laplace_func(i / np.sqrt(self.sigma_sqr))
            index += 1
        p_i_another = n_i / n

        res_sum = 0
        for i in range(self.bins):
            if p_i[i] == 0:
                raise ValueError("p_i[i] РАВЕН НУЛЮ, ТАК НЕЛЬЗЯ!!!")
            res_sum += ((p_i[i] - p_i_another[i]) ** 2) / p_i[i]
        statistic = n * res_sum

        count_unknown_params = 1
        chi_min = chi(self.bins + 1 - count_unknown_params, alpha1 / 2)
        chi_max = chi(self.bins + 1 - count_unknown_params, 1 - alpha1 / 2)

        self.task8_statistic = statistic
        self.task8_chi_min = chi_min
        self.task8_chi_max = chi_max

        return self

    def func_view(self, accuracy=2, epsilon=0.01):
        if self.theta is None:
            return None
        _result = ""
        _reg_index = 0
        for _regressor in self.__reg_types:
            if isinstance(_regressor, Polynomial):
                if self.theta[_reg_index] != 0:
                    _result += f"({round(self.theta[_reg_index], accuracy)}) + "
                    _reg_index += 1
                for _i in range(1, _regressor.max_coefficient + 1):
                    if -epsilon < self.theta[_reg_index] < epsilon:
                        _reg_index += 1
                        continue
                    _result += f"({round(self.theta[_reg_index], accuracy)})x**{_i} + "
                    _reg_index += 1
            elif isinstance(_regressor, Exponential):
                for _i in range(1, _regressor.max_coefficient + 1):
                    if -epsilon < self.theta[_reg_index] < epsilon:
                        _reg_index += 1
                        continue
                    _result += f"({round(self.theta[_reg_index], accuracy)})e**({_i}x) + "
                    _reg_index += 1
            elif isinstance(_regressor, Sinusoid):
                for _i in range(1, _regressor.max_coefficient + 1):
                    if -epsilon < self.theta[_reg_index] < epsilon:
                        _reg_index += 1
                        continue
                    _result += f"({round(self.theta[_reg_index], accuracy)})sin({_i}x) + "
                    _reg_index += 1
            elif isinstance(_regressor, CoSinusoid):
                for _i in range(1, _regressor.max_coefficient + 1):
                    if -epsilon < self.theta[_reg_index] < epsilon:
                        _reg_index += 1
                        continue
                    _result += f"({round(self.theta[_reg_index], accuracy)})cos({_i}x) + "
                    _reg_index += 1
        return _result[:-3]

    def show(self):
        plt.plot(self.x, self.Y, "o", color="black")
        x = self.x_linspace
        plt.plot(x, self.y_values, color="red")
        plt.plot(x, self.func_lower_alpha1, color="pink")
        plt.plot(x, self.func_upper_alpha1, color="pink")
        plt.plot(x, self.func_lower_alpha2, color="orange")
        plt.plot(x, self.func_upper_alpha2, color="orange")

        bins = self.bins
        fig, ax = plt.subplots()
        ax.hist(self.err_vector, bins=bins)

        plt.show()

