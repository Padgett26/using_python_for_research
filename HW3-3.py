import os
import numpy as np
import random
import scipy.stats as ss
import pandas as pd
import sklearn.preprocessing as sp
import sklearn.decomposition as sd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages


data_dir = "./data"


# random.seed(123)  # do not change
random.seed(1)  # do not change


def accuracy(predictions, outcomes):
    p = 0
    for i in range(len(predictions)):
        p += 1 if predictions[i] == outcomes[i] else 0
    percent = (p / len(predictions)) * 100
    return percent


def accuracy_x_single(predictions, outcomes):
    p = 0
    for i in range(len(outcomes)):
        p += 1 if predictions == outcomes[i] else 0
    percent = (p / len(outcomes)) * 100
    return percent


def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode


def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))


def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]


def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]


wine = pd.read_csv(os.path.join(data_dir, "wine.csv"), index_col=0)

numeric_data = np.zeros((len(wine), 12))
data = np.zeros((len(wine), 14))

for w in range(len(wine)):
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality, color, high_quality = \
        wine.iloc[w]

    numeric_data[w] = [
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
        1 if color == "red" else 0
    ]

    data[w] = [
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol,
        quality,
        1 if color == "red" else 0,
        high_quality
    ]

# print(numeric_data)


data = pd.DataFrame(data, columns=["fixed_acidity", "volatile_acidity", "citric_acid",
                                   "residual_sugar", "chlorides", "free_sulfur_dioxide",
                                   "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol",
                                   "quality", "color", "high_quality"])

scaled_data = sp.StandardScaler().fit(numeric_data)
numeric_data = pd.DataFrame(scaled_data.transform(numeric_data), columns=["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "is_red"])
# print(scaled_data.mean_)
# print(scaled_data.scale_)

pca = sd.PCA(n_components=2)
principal_components = pca.fit_transform(numeric_data)

print("principal components shape: " + str(principal_components.shape))

observation_colormap = ListedColormap(['red', 'blue'])
x = principal_components[:, 0]
y = principal_components[:, 1]

plt.title("Principal Components of Wine")
plt.scatter(x, y, alpha=0.2, c=data['high_quality'], cmap=observation_colormap, edgecolors='none')
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
# plt.show()


c = np.random.randint(0, 2, 1000)
d = np.random.randint(0, 2, 1000)

print("accuracy x y: " + str(accuracy(c, d)))

low_quality = accuracy_x_single(0, data['high_quality'])
high_quality = accuracy_x_single(1, data['high_quality'])

print("accuracy low quality: " + str(low_quality) + "%")
print("accuracy high quality: " + str(high_quality) + "%")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(numeric_data, data['high_quality'])

library_predictions = knn.predict(numeric_data)
knnP = accuracy(library_predictions, data['high_quality'])

print("knn predict: " + str(knnP))

n_rows = data.shape[0]
selection = random.sample(range(n_rows), 10)

print("selected sample rows: ", selection)

predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
# print("training indices: ", training_indices)

outcomes = np.array(data["high_quality"])

my_predictions = np.zeros(10)
hq = np.zeros(10)

t = 0
for p in predictors[selection]:
    my_predictions[t] = knn_predict(p, predictors[training_indices, :], outcomes[training_indices], k=5)
    t += 1

r = 0
for h in selection:
    hq[r] = data['high_quality'][h]
    r += 1

percentage = accuracy(my_predictions, hq)
print(percentage)

red = 0
for i in range(len(data)):
    red += data['color'][i]

print("red: " + str(red))
