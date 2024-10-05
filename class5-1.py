import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ss.norm.rvs(loc=0, scale=1, size=n)
X = np.stack([x_1, x_2], axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$y$")
# fig.show()

lm = LinearRegression(fit_intercept=True)
lm.fit(X, y)
print("Intercept: ", lm.intercept_)
print("Beta 1: ", lm.coef_[0])
print("Beta 2: ", lm.coef_[1])

X_0 = np.array([2, 4])
pred = lm.predict(X_0.reshape(1, -1))
score = lm.score(X, y)
print("Prediction: ", pred)
print("Prediction score: ", score)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
lm = LinearRegression(fit_intercept=True)
lm.fit(X_train, y_train)
lm.score(X_test, y_test)

# plt.plot(x, y, "o", ms=5)
# xx = np.array([0, 10])
# plt.plot(xx, beta_0 + beta_1 * xx)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

def compute_rss(y_estimate, y):
    return sum(np.power(y-y_estimate, 2))

def estimate_y(x, b_0, b_1):
    return b_0 + b_1 * x

# rss = compute_rss(estimate_y(x, beta_0, beta_1), y)

# rss = []
# slopes = np.arange(-10, 15, 0.001)
# for slope in slopes:
#     rss.append(np.sum((y - beta_0 - slope * x)**2))

# ind_min = np.argmin(rss)

# print("Estimate for the slope: ", slopes[ind_min])

# plt.figure()
# plt.plot(slopes, rss)
# plt.xlabel("Slope")
# plt.ylabel("RSS")
# plt.show()

# X = sm.add_constant(x)
# mod = sm.OLS(y, X)
# est = mod.fit()
# print(est.summary())
