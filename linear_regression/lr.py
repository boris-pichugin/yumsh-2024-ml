import random


def main() -> None:
    size = 10000
    n = 10
    points = [random_point(n) for _ in range(size)]
    targets = [linear(x) for x in points]


def random_point(n) -> list[float]:
    return [random.random() * 2.0 - 1.0 for _ in range(n)]


def linear(x: list[float]) -> float:
    return sum((i + 1) * x[i] for i in range(len(x))) + random.gauss(0, 0.01)
