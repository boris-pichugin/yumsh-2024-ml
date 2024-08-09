import matplotlib.pyplot as plt
import random as rnd

rnd.seed(1)

N = 8
roots = [rnd.random() * 100 - 50 for _ in range(N)]
n = 0


def f(x: float) -> float:
    global n
    n += 1
    p = 1.0
    for x_i in roots:
        p *= x - x_i
    return p


x0 = min(roots)
x1 = max(roots)
d = x1 - x0
x0 -= 0.1 * d
x1 += 0.1 * d

X = [x0 + i * (x1 - x0) / 10000 for i in range(10000)]
Y = [f(x) for x in X]

plt.plot(X, Y)
plt.show()
