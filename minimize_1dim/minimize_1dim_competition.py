import math
import random
from typing import Callable

import matplotlib.pyplot as plt
import random as rnd


def main():
    global n

    n = 0
    min_arg = minimize_1d(f)
    print(min_arg)
    print(f(min_arg))
    print(n)

    X = [x0 + i * (x1 - x0) / 10000 for i in range(10000)]
    Y = [f(x) for x in X]

    plt.plot([min_arg, min_arg], [min(Y), max(Y)])
    plt.plot(X, Y)
    plt.show()


rnd.seed(42)

N = 16
n = 0
roots = [rnd.random() * 20 for _ in range(N)]


def f(x: float) -> float:
    global n
    n += 1
    p = 1.0
    for x_i in roots:
        p *= x - x_i
    return p + math.sin(x * x) * 1e15


x0 = min(roots)
x1 = max(roots)
d = x1 - x0
x0 -= 0.1 * d
x1 += 0.1 * d


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


def minimize_1d_from_point(f: Callable[[float], float], x0: float) -> float:
    h = 0.001
    f_x0 = f(x0)

    x1 = x0 + h
    f_x1 = f(x1)

    if f_x0 < f_x1:
        x0, x1 = x1, x0
        f_x0, f_x1 = f_x1, f_x0
        h = -h

    while True:
        x2 = x1 + h
        f_x2 = f(x2)
        if f_x2 < f_x1:
            h *= 1.2
            x0, x1 = x1, x2
            f_x0, f_x1 = f_x1, f_x2
        elif x0 < x2:
            return golden_sect(f, x0, x2)
        else:
            return golden_sect(f, x2, x0)


EPS = 1e-6


def golden_sect(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Алгоритм золотого сечения.
    """
    lmbd = (3 - 5 ** 0.5) / 2
    x1 = a * (1 - lmbd) + b * lmbd
    x2 = a * lmbd + b * (1 - lmbd)
    f1 = f(x1)
    f2 = f(x2)

    while EPS <= (b - a):
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a * (1 - lmbd) + b * lmbd
            f2 = f1
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            x2 = a * lmbd + b * (1 - lmbd)
            f1 = f2
            f2 = f(x2)

    return (a + b) / 2


if __name__ == '__main__':
    main()
