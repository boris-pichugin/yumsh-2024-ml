import random
from typing import Callable

import function
import matplotlib.pyplot as plt


def main() -> None:
    min_x = minimize_2d(function.f)
    print(min_x)
    print(function.f(min_x))

    function.draw_2d_function(function.f)
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


def minimize_2d_from_point(f: Callable[[list[float]], float], x: list[float]) -> list[float]:
    pass


if __name__ == '__main__':
    main()
