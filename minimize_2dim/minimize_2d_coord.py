import random
from typing import Callable

import matplotlib.pyplot as plt

import minimize_1dim.minimize_1d
import minimize_2dim.function_2d as fun2d


def main() -> None:
    fun2d.draw_2d_function(fun2d.f)

    min_x = minimize_2d(fun2d.f)
    print(min_x)
    print(fun2d.f(min_x))

    plt.show()


def minimize_2d(f: Callable[[list[float]], float], tries: int = 10, steps: int = 100) -> list[float]:
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
    def f1(z: float):
        return f([z, x[1]])

    def f2(z: float):
        return f([x[0], z])

    ff = [f1, f2]
    path = [x.copy()]

    for i in range(steps):
        direction = i % 2
        f1 = ff[direction]
        x[direction] = minimize_1dim.minimize_1d.minimize_1d_from_point(f1, x[direction], steps=100)
        path.append(x.copy())

    fun2d.draw_path(path)

    return x


if __name__ == '__main__':
    main()
