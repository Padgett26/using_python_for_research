import random

random.seed(1)  # Fixes the see of the random number generator.


# 3a
def moving_window_average(x, n_neighbors=1):
    result = []
    n = len(x)
    width = n_neighbors * 2 + 1
    for i in range(n):
        window = []
        for j in range(width):
            r = i - n_neighbors + j
            k = r if n > r >= 0 else i
            window.append(x[k])
        result.append(sum(window) / width)
    return result


x = [0, 10, 5, 3, 1, 5]
example = moving_window_average(x, 1)


# print(example)


# 3b
def rand():
    r = random.uniform(0, 1)
    return r


Y = []
temp = []
for i in range(1000):
    temp.append(rand())

Y.append(temp)

for i in range(1, 10):
    Y.append(moving_window_average(Y[0], i))
    print("neighbors: " + str(i) + " = " + str(Y[i]))

# 3c
ranges = []
for i in range(len(Y)):
    sorted_list = sorted(Y[i])
    diff = sorted_list[len(Y) - 1] - sorted_list[0]
    ranges.append(diff)
    print("range with " + str(i) + " neighbors: " + str(diff))

write_up = "\nThe resulting ranges seem to be in the shape of a sign wave with decreasing amplitude and increasing frequency,\ncolasping to a range of about 0.0424989 (at 9 neighbors). The results seem to fluctuate, collasping closer to the global average."
print(write_up)
