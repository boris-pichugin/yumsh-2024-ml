import math
import random
from typing import Callable

import matplotlib.pyplot as plt
import seaborn as sns

random.seed(2)
# Число локальных минимумов
N = 5
# ТОчки локальных минимумов
MX = [[random.random(), random.random()] for _ in range(N)]
# Значения в точках локальных минимумов
MZ = [random.random() * 0.3 for _ in range(N)]
# Коэффициенты растяжения по осям в окрестности минимумов
MK = [[[random.random(), random.random()], [random.random(), random.random()]] for _ in range(N)]
# Масштаб графика
M = 200

for i in range(N):
    a = (MK[i][0][1] + MK[i][1][0]) / 6
    MK[i][0][1] = MK[i][1][0] = a


def f(x: list[float]) -> float:
    return min(MZ[k] + dist(x, MX[k], MK[k]) for k in range(N)) + 0.005 * math.sin(x[0] * 100) * math.sin(x[1] * 100)
    # return min(MZ[k] + dist(x, MX[k], MK[k]) for k in range(N)) + math.sin(x[0]*x[1]*5)


def f2(x: list[float]) -> float:
    return abs(math.sin(x[0] * math.pi) - x[1]) + 0.01 * x[0]


def f3(x: list[float]) -> float:
    return (math.sin(x[0] * math.pi) - x[1]) ** 2 + 0.01 * x[0]


def dist(x: list[float], y: list[float], c: list[list[float]]) -> float:
    s = 0
    for i in range(len(x)):
        d_i = x[i] - y[i]
        for j in range(len(y)):
            d_j = x[j] - y[j]
            s += c[i][j] * d_i * d_j
    # return sum(c[k] * (x[k] - y[k]) ** 2 for k in range(len(x)))
    return s


def draw_2d_function(f: Callable[[list[float]], float]) -> None:
    z = [[f([i / M, j / M]) for i in range(M + 1)] for j in range(M + 1)]
    labels = [f"{i / M:.2f}" if i % 10 == 0 else None for i in range(M + 1)]

    sns.heatmap(z, xticklabels=labels, yticklabels=labels)
    min_z = min(min(u) for u in z)
    max_z = max(max(u) for u in z)
    num_levels = 50
    levels = [min_z + (max_z - min_z) * i / num_levels for i in range(0, num_levels + 1)]
    plt.contour(z, colors=[(0.6, 0.6, 0.6)], levels=levels)


def draw_path(path: list[list[float]]) -> None:
    color = (
        0.7 + 0.3 * random.random(),
        0.7 + 0.3 * random.random(),
        0.7 + 0.3 * random.random()
    )
    x = [p[0] * M for p in path]
    y = [p[1] * M for p in path]
    plt.plot(x, y, color=color)


if __name__ == '__main__':
    draw_2d_function(f)
    plt.show()
