from typing import Callable

EPS = 1e-6
Y = [0.1, 2, 3, 4, 7, 10, 11, 22]


# https://github.com/boris-pichugin/yumsh-2024-ml

def main():
    global n
    answer = sum(Y) / len(Y)

    n = 0
    min_arg = minimize_func_1(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)

    n = 0
    min_arg = minimize_func_2(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)

    n = 0
    min_arg = minimize_func_3(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)


def minimize_func_1(f: Callable[[float], float], a: float, b: float) -> float:
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


def minimize_func_2(f: Callable[[float], float], a: float, b: float) -> float:
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


def minimize_func_3(f: Callable[[float], float], a: float, b: float) -> float:
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


def minimize_func_4(f: Callable[[float], float], a: float, b: float) -> float:
    """
    Алгоритм парабол.
    """
    lmbd = (3 - 5 ** 0.5) / 2
    x1 = a * (1 - lmbd) + b * lmbd
    fa = f(a)
    fb = f(b)
    f1 = f(x1)
    x2 = min_of_parabol(a, fa, b, fb, x1, f1)
    f2 = f(x2)
    if x1 < x2:
        x1, x2 = x2, x1
        f1, f2 = f2, f1

    while EPS <= (b - a):
        if f1 < f2:
            b = x2
            fb = f2
            f2 = f1
            x2 = x1
            x1 = min_of_parabol(a, fa, b, fb, x2, f2)
            f1 = f(x1)
        else:
            a = x1
            fa = f1
            f1 = f2
            x1 = x2
            x2 = min_of_parabol(a, fa, b, fb, x1, f1)
            f2 = f(x2)
        if x1 < x2:
            x1, x2 = x2, x1
            f1, f2 = f2, f1

    return (a + b) / 2


n = 0


def f(x: float) -> float:
    global n
    n += 1
    return sum((x - y) ** 2 for y in Y) / len(Y)


if __name__ == '__main__':
    main()
