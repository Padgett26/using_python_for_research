# 2a
import math
import random

a = math.pi / 4
print(a)

# 2b
random.seed(1)  # Fixes the see of the random number generator.


def rand():
    r = random.uniform(-1, 1)
    return r


print(rand())


# 2c
def distance(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


x = [0, 0]
y = [1, 1]
print(distance(x, y))


# 2d
def in_circle(x, origin=[0, 0]):
    return distance(x, origin) < 1


# x = [0.5, 0.5]
x = [1, 1]
o = [0, 0]

print(in_circle(x, o))

# 2e
R = 10000
inside = []
for i in range(R):
    x = [rand(), rand()]
    o = [0, 0]
    inside.append(in_circle(x, o))

num = 0
for i in range(R):
    if inside[i]:
        num += 1

b = num / R
print(b)

# 2f
print(abs(a - b))  # a from section 2a (math.pi / 4) and b from section 2e (num / R)
