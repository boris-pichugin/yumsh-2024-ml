import random
from typing import Callable

import matplotlib.pyplot as plt

import minimize_1dim.function_1d as fun1d
from minimize_1dim.minimize_1d import minimize_1d_from_point

EPS = 1e-6


def main():
    fun1d.n = 0
    min_arg = minimize_1d(fun1d.f)
    print(min_arg)
    print(fun1d.f(min_arg))
    print(fun1d.n)

    fun1d.draw_1d_function(fun1d.f)
    fun1d.draw_vertical(min_arg)
    plt.show()


def minimize_1d(f: Callable[[float], float]) -> float:
    min_x = 0
    min_f = f(min_x)
    for i in range(7):
        x = random.gauss(0, 100)
        x = minimize_1d_from_point(f, x)
        f_x = f(x)
        if f_x < min_f:
            min_x = x
            min_f = f_x
    return min_x


if __name__ == '__main__':
    main()
