from typing import Callable

EPS = 1e-6
Y = [0.1, 2, 3, 4, 7, 10, 11, 22]


def main():
    answer = sum(Y) / len(Y)
    min_arg = minimize_func(f, min(Y), max(Y))
    print(abs(answer - min_arg))
    print(n)


def minimize_func(f: Callable[[float], float], a: float, b: float) -> float:
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


n = 0


def f(x: float) -> float:
    global n
    n += 1
    return sum((x - y) ** 2 for y in Y) / len(Y)


if __name__ == '__main__':
    main()
