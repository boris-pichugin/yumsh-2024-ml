import random
from typing import Callable

import matplotlib.pyplot as plt

import minimize_2dim.function_2d as fun2d


def main() -> None:
    fun2d.draw_2d_function(fun2d.f)

    min_x = minimize_2d(fun2d.f)
    print(min_x)
    print(fun2d.f(min_x))

    plt.show()


def minimize_2d(f: Callable[[list[float]], float], tries: int = 20, steps: int = 10000) -> list[float]:
    min_x = [0.5, 0.5]
    min_f = f(min_x)
    for i in range(tries):
        x = [random.random(), random.random()]
        x = minimize_2d_from_point(f, x, steps)
        f_x = f(x)
        if f_x < min_f:
            min_x = x
            min_f = f_x
    return min_x


def minimize_2d_from_point(f: Callable[[list[float]], float], x: list[float], steps: int = 1000):
    path = [x.copy()]

    for i in range(steps):
        g = grad(f, x)
        x = plus_xy(x, -0.01, g)
        path.append(x.copy())

    fun2d.draw_path(path)

    return x


def grad(f: Callable[[list[float]], float], x: list[float]) -> list[float]:
    h = 1e-6
    f_x = f(x)
    return [(f(plus_xh(x, h, i)) - f_x) / h for i in range(len(x))]


def plus_xh(x: list[float], h: float, i: int) -> list[float]:
    xh = x.copy()
    xh[i] += h
    return xh


def plus_xy(x: list[float], a: float, y: list[float]) -> list[float]:
    for i in range(len(x)):
        x[i] = x[i] + a * y[i]
    return x


if __name__ == '__main__':
    main()
