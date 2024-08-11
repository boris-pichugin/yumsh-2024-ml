from typing import Callable

import seaborn as sns
import random
import matplotlib.pyplot as plt

random.seed(3)
N = 5
MX = [[random.random(), random.random()] for _ in range(N)]
MZ = [random.random() * 0.3 for _ in range(N)]
MK = [[random.random(), random.random()] for _ in range(N)]


def f(x: list[float]) -> float:
    return min(MZ[k] + dist(x, MX[k], MK[k]) for k in range(N))
    # return min(MZ[k] + dist(x, MX[k], MK[k]) for k in range(N)) + math.sin(x[0]*x[1]*5)


def dist(x: list[float], y: list[float], c: list[float]) -> float:
    return sum(c[k] * (x[k] - y[k]) ** 2 for k in range(len(x)))


def draw_2d_function(f: Callable[[list[float]], float]) -> None:
    M = 200
    z = [[f([i / M, j / M]) for i in range(M)] for j in range(M)]

    sns.heatmap(z)
    min_x = min(min(u) for u in z)
    max_x = max(max(u) for u in z)
    num_levels = 50
    levels = [min_x + (max_x - min_x) * i / num_levels for i in range(0, num_levels + 1)]
    plt.contour(z, colors="white", levels=levels)


if __name__ == '__main__':
    draw_2d_function(f)
    plt.show()
