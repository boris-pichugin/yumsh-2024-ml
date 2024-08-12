import random
from typing import Callable

import minimize_2dim.function as function
import matplotlib.pyplot as plt

import minimize_1dim.minimize_1dim_competition as dim1


def main() -> None:
    function.draw_2d_function(function.f)

    min_x = minimize_2d(function.f)
    print(min_x)
    print(function.f(min_x))

    plt.show()


def minimize_2d(f: Callable[[list[float]], float]) -> list[float]:
    min_x = [0.5, 0.5]
    min_f = f(min_x)
    for i in range(7):
        x = [random.random(), random.random()]
        x = minimize_2d_from_point(f, x)
        f_x = f(x)
        if f_x < min_f:
            min_x = x
            min_f = f_x
    return min_x


def minimize_2d_from_point(f: Callable[[list[float]], float], x: list[float]):
    def f1(z: float):
        return f([z, x[1]])

    def f2(z: float):
        return f([x[0], z])

    ff = [f1, f2]
    path = [x.copy()]

    for i in range(100):
        direction = i % 2
        f1 = ff[direction]
        x[direction] = dim1.minimize_1d_from_point(f1, x[direction])
        path.append(x.copy())

    function.draw_path(path)

    return x


if __name__ == '__main__':
    main()
