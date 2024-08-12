from typing import Callable

EPS = 1e-6
Y = [0.1, 2, 3, 4, 7, 10, 11, 22]


# https://github.com/boris-pichugin/yumsh-2024-ml

def main():
    global n
    answer = sum(Y) / len(Y)

    n = 0
    min_arg = minimize_scan(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)

    n = 0
    min_arg = minimize_triple_split(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)

    n = 0
    min_arg = minimize_golden_sect(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)

    n = 0
    min_arg = minimize_parabolic(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)

    n = 0
    min_arg = minimize_grad(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)


def minimize_1d_from_point(f: Callable[[float], float], x0: float, steps: int = 1000, accuracy: float = 1e-6) -> float:
    h = 0.001
    f_x0 = f(x0)

    x1 = x0 + h
    f_x1 = f(x1)

    if f_x0 < f_x1:
        x0, x1 = x1, x0
        f_x0, f_x1 = f_x1, f_x0
        h = -h

    for _ in range(steps):
        x2 = x1 + h
        f_x2 = f(x2)
        if f_x2 < f_x1:
            h *= 1.2
            x0, x1 = x1, x2
            f_x0, f_x1 = f_x1, f_x2
        elif x0 < x2:
            return minimize_golden_sect(f, x0, x2, accuracy=accuracy)
        else:
            return minimize_golden_sect(f, x2, x0, accuracy=accuracy)

    return x1


def minimize_scan(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Алгоритм линейного поиска.
    """
    x = a
    fx = f(x)
    h = (b - a) / 1000
    while True:
        x1 = min(max(x + h, a), b)
        fx1 = f(x1)
        if x1 == a or x1 == b or fx < fx1:
            h /= -10
            if abs(h) < EPS:
                return x if fx < fx1 else x1
        x = x1
        fx = fx1


def minimize_triple_split(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Алгоритм деления отрезка на три части.
    """
    while EPS <= (b - a):
        x1 = a * 2 / 3 + b / 3
        x2 = a / 3 + b * 2 / 3
        f1 = f(x1)
        f2 = f(x2)
        if f1 < f2:
            b = x2
        else:
            a = x1
    return (a + b) / 2


def minimize_golden_sect(f: Callable[[float], float], a: float, b: float, accuracy: float = EPS) -> float:
    """
    Алгоритм золотого сечения.
    """
    lmbd = (3 - 5 ** 0.5) / 2
    x1 = a * (1 - lmbd) + b * lmbd
    x2 = a * lmbd + b * (1 - lmbd)
    f1 = f(x1)
    f2 = f(x2)

    while accuracy <= (b - a):
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


def minimize_parabolic(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Алгоритм парабол.
    """
    lmbd = (3 - 5 ** 0.5) / 2
    x1 = a * (1 - lmbd) + b * lmbd
    fa = f(a)
    fb = f(b)
    f1 = f(x1)
    x2 = min_of_parabola(a, fa, b, fb, x1, f1)
    f2 = f(x2)
    if x1 > x2:
        x1, x2 = x2, x1
        f1, f2 = f2, f1

    while EPS <= (b - a):
        if f1 < f2:
            b = x2
            fb = f2
            f2 = f1
            x2 = x1
            x1 = min_of_parabola(a, fa, b, fb, x2, f2)
            f1 = f(x1)
        else:
            a = x1
            fa = f1
            f1 = f2
            x1 = x2
            x2 = min_of_parabola(a, fa, b, fb, x1, f1)
            f2 = f(x2)
        if x1 > x2:
            x1, x2 = x2, x1
            f1, f2 = f2, f1

    return (a + b) / 2


def min_of_parabola(a, fa, b, fb, x, fx):
    if fx >= fa + (fb - fa) * (x - a) / (b - a):
        return golden_sect_fallback(a, b, x)

    xn = (x - a) / (b - a)
    k2 = (fx - fa - (fb - fa) * xn) / (xn * xn - xn)
    if k2 < EPS:
        return golden_sect_fallback(a, b, x)

    k1 = fb - fa - k2
    x0n = -k1 / (2 * k2)
    x0 = a + x0n * (b - a)
    if a < x0 < b:
        return x0

    return golden_sect_fallback(a, b, x)


def golden_sect_fallback(a, b, x):
    lmbd = (3 - 5 ** 0.5) / 2
    p1 = a * lmbd + b * (1 - lmbd)
    p2 = a * (1 - lmbd) + b * lmbd
    if abs(x - p1) < abs(x - p2):
        return p2
    else:
        return p1


def minimize_grad(f: Callable[[float], float], a: float, b: float, steps: int = 1000, accuracy: float = EPS) -> float:
    h = 0.0001
    l = 1
    moment = 0.5

    x = (a + b) / 2

    for _ in range(steps):
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        x1 = x - l * dfx
        x2 = moment * x + (1 - moment) * x1
        x2 = max(min(x2, b), a)
        if abs(x2 - x) < accuracy:
            return x2
        x = x2

    return x


n = 0


def f(x: float) -> float:
    global n
    n += 1
    return sum((x - y) ** 2 for y in Y) / len(Y)


if __name__ == '__main__':
    main()
