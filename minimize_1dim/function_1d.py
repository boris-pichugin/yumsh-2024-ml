import math
import random as rnd
from typing import Callable

from matplotlib import pyplot as plt

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


def draw_1d_function(f: Callable[[float], float]) -> None:
    x0 = min(roots)
    x1 = max(roots)
    d = x1 - x0
    x0 -= 0.1 * d
    x1 += 0.1 * d

    x = [x0 + i * (x1 - x0) / 10000 for i in range(10000)]
    y = [f(x) for x in x]

    plt.plot(x, y)


def draw_vertical(x: float) -> None:
    plt.plot([x, x], plt.gca().get_ylim())
