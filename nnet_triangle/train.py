import math
import random
from typing import Callable

from matplotlib import pyplot as plt

import minimize_2dim.function_2d as fun2d
from minimize_2dim.function_2d import draw_2d_function
from nnet_triangle.oracle import oracle

pack_start = 0
pack_size = 20


def main():
    n = 10000
    points = [rnd_point() for _ in range(n)]
    targets = [oracle(p) for p in points]

    batch_size = 100

    def loss(w: list[float], s: int) -> float:
        s0 = (s * batch_size) % n
        s1 = s0 + batch_size
        ls = 0.0
        for i in range(s0, s1):
            ls += (nnet(points[i], w) - targets[i]) ** 2
        ls /= 20
        for w_i in w:
            ls += 0.0000001 * w_i ** 2
        return ls

    w = [random.random() - 0.5 for _ in range(13)]
    w = minimize_2d_from_point(
        loss,
        w,
        steps=100000,
        lr=0.02,
        beta_1=0.85
    )

    def final_nnet(x: list[float]) -> float:
        return nnet(x, w)

    draw_2d_function(final_nnet, -1, -1, 1, 1, False)
    plt.show()


def nnet(x: list[float], w: list[float]) -> float:
    n1 = sigma(w[0] + w[1] * x[0] + w[2] * x[1])
    n2 = sigma(w[3] + w[4] * x[0] + w[5] * x[1])
    n3 = sigma(w[6] + w[7] * x[0] + w[8] * x[1])
    return sigma(w[9] + w[10] * n1 + w[11] * n2 + w[12] * n3)


def sigma(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def rnd_point() -> list[float]:
    return [random.random() * 2.0 - 1.0, random.random() * 2.0 - 1.0]


def minimize_2d_from_point(
        f: Callable[[list[float], int], float],
        x: list[float],
        steps: int = 1000,
        lr: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        epsilon: float = 0.01
):
    path = [x.copy()]

    v = [0.0 for _ in range(len(x))]
    q = [1.0 for _ in range(len(x))]
    for s in range(steps):
        g = grad(f, x, s)
        for i in range(len(x)):
            v[i] = beta_1 * v[i] + (1 - beta_1) * g[i]
            q[i] = beta_2 * q[i] + (1 - beta_2) * (v[i] ** 2)
            x[i] = x[i] - lr * v[i] / (epsilon + q[i] ** 0.5)

        if (s + 1) % 100 == 0:
            print(f"{s:10d} {f(x, s)}")
        path.append(x.copy())

    fun2d.draw_path(path)

    return x


def grad(f: Callable[[list[float], int], float], x: list[float], s: int) -> list[float]:
    h = 1e-6
    f_x = f(x, s)
    return [(f(plus_xh(x, h, i), s) - f_x) / h for i in range(len(x))]


def plus_xh(x: list[float], h: float, i: int) -> list[float]:
    xh = x.copy()
    xh[i] += h
    return xh


if __name__ == '__main__':
    main()
