import numpy as np
import openpyxl
import warnings
import matplotlib.pyplot as plt
import matplotlib

import regression_model as rg

VARIANT = 48


def work(model):
    print(f"Матрица X:\n{model.X.T}\n")
    print(f"Матрица X^T:\n{model.X}\n")
    print(f"Матрица (X^T * X):\n{model.X_T_X}\n")
    print(f"Матрица (X^T * X)^-1:\n{model.X_inv}\n")
    print(f"Значения alpha_k:\n{model.alpha_k}\n\n\n")

    print(f"КУРСОВАЯ РАБОТА\n")
    print(f"1) МНК оценка коэффициентов theta:\n{model.theta}\n")
    print(f"Вид функции phi(x; theta^):\n{model.func_view()}\n")
    print(f"2) Центральные доверительные интервалы для theta уровней надёжности:")
    print(f" а) 0.95:")
    print(f"{model.task2_1}\n")
    print(f" б) 0.99:")
    print(f"{model.task2_2}\n")
    print(f"3) МП оценка дисперсии случайной ошибки:\n{model.sigma_sqr}\n")
    print(f"4) Центральные доверительные интервалы для дисперсии случайной ошибки уровней надёжности:")
    print(f" а) 0.95:")
    print(f"{model.task4_1}\n")
    print(f" б) 0.99:")
    print(f"{model.task4_2}\n")
    print(f"5) Центральные доверительные интервалы для полезного сигнала:")
    print(f"alpha(x) = {model.alpha_x_view}")
    print(f" a) 0.95:")
    print(f"{model.task5_1}")
    print()
    print(f" б) 0.99:")
    print(f"{model.task5_2}")
    print()
    print(f"6) см. графики\n")
    print(f"7) см. гистограмму")
    print(f"Вектор остатков регрессии:\n{model.err_vector}\n")
    print(f"8) Проверка гипотезы")
    print(f"{model.task8_chi_min} < {model.task8_statistic} < {model.task8_chi_max}")
    if model.task8_chi_min < model.task8_statistic < model.task8_chi_max:
        print(f"Гипотеза принимается.")
    else:
        print(f"Гипотеза отвергается...")
    model.show()


def main():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb = openpyxl.load_workbook(f"Data{VARIANT}.xlsx")
    sheet = wb.active
    Y_values = np.array([sheet.cell(row=i, column=1).value for i in range(1, len(sheet["A"]) + 1)], dtype=np.float64)
    x_values = np.array([-1 + (k - 1) / 20 for k in range(1, 41 + 1)], dtype=np.float64)

    model = rg.RegressionModel(x_values, Y_values, [rg.Polynomial(2)]).calculate()
    work(model)


if __name__ == "__main__":
    main()
