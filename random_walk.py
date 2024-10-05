import numpy as np
import matplotlib.pyplot as plt

num_walks = 1000
X_0 = np.array([[0], [0]])
delta_X = np.random.normal(0, 1, (2, num_walks))
X = np.concatenate((X_0, np.cumsum(delta_X, axis=1)), axis=1)
plt.plot(X[0], X[1], "ro-")
plt.show()
