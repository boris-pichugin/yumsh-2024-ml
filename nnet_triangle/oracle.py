import random

from matplotlib import pyplot as plt

from minimize_2dim.function_2d import draw_2d_function

random.seed(42)

A = (random.random() * 2 - 1, random.random() * 2 - 1)
B = (random.random() * 2 - 1, random.random() * 2 - 1)
C = (random.random() * 2 - 1, random.random() * 2 - 1)


def oracle(x: list[float]) -> float:
    if (_is_one_side(x, A, B, C)
            and _is_one_side(x, A, C, B)
            and _is_one_side(x, B, C, A)):
        return 1
    else:
        return 0


def _is_one_side(x: list[float], A: tuple[float, float], B: tuple[float, float], C: tuple[float, float]) -> bool:
    a = B[1] - A[1]
    b = -(B[0] - A[0])
    c = -a * A[0] - b * A[1]
    return (a * x[0] + b * x[1] + c) * (a * C[0] + b * C[1] + c) > 0


if __name__ == '__main__':
    draw_2d_function(oracle, -1, -1, 1, 1, False)
    plt.show()
