import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

h = 1.5
sd1 = 1
sd2 = 1.5
n = 1000

def gen_data(n, h, sd1, sd2):
    x1 = ss.norm.rvs(-h, sd1, n)
    y1 = ss.norm.rvs(0, sd1, n)
    x2 = ss.norm.rvs(h, sd2, n)
    y2= ss.norm.rvs(0, sd2, n)
    return (x1, y1, x2, y2)


def plot_data(x1, y1, x2, y2):
    plt.figure()
    plt.plot(x1, y1, "o", ms=2)
    plt.plot(x2, y2, "o", ms=2)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    plt.show()

# (x1, y1, x2, y2) = gen_data(1000, 1.5, 1, 1.5)
# plot_data(x1, y1, x2, y2)

def prob_to_odds(p):
    if p <= 0 or p >= 1:
        print("Probabilities must be between 0 and 1.")
    return p / (1-p)

# print(prob_to_odds(0.8))

def plot_probs(ax, clf, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1))
    Z = probs[:,class_no]
    Z = Z.reshape(xx1.shape)
    CS = ax.contourf(xx1, xx2, Z)
    cbar = plt.colorbar(CS)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")

clf = LogisticRegression()
(x1, y1, x2, y2) = gen_data(n, h, sd1, sd2)

X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))
y = np.hstack((np.repeat(1, n), np.repeat(2, n)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

prob = clf.predict_proba(np.array([-2, 0]).reshape(1, -1))
predict = clf.predict(np.array([-2, 0]).reshape(1, -1))

print("Score: ", score)
print("Probability the -2, 0 point will be in class 1, and class 2: ", prob)
print("Prediction of which class the point will be a part of: ", predict)

plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0)
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1)
plt.title("Pred. prob for class 2")
plt.show()

