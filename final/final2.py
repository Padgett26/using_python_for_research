# Needed imports
import datetime
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

data_dir = "./data"

# start process timer
start_time = datetime.datetime.now()

# Load data from files - training data
df_train_time_series = pd.read_csv(os.path.join(data_dir, "train_time_series.csv"))
df_train_labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))

# Load data from files - test data
df_test_time_series = pd.read_csv(os.path.join(data_dir, "test_time_series.csv"))
df_test_labels = pd.read_csv(os.path.join(data_dir, "test_labels_input.csv"))


# functional processes
def train_data(train_ts, train_l):
    """Grabbing the training data imported from files, and forming it into table that I can work with."""
    train_table = []
    for i in range(len(train_l)):
        x = 0
        y = 0
        z = 0
        ind = None
        for j in range(len(train_ts)):
            ind = (
                train_l.orig_index.iloc[i]
                if train_ts.timestamp.iloc[j] == train_l.timestamp.iloc[i]
                else None
            )
            if ind:
                x = train_ts.x.iloc[j]
                y = train_ts.y.iloc[j]
                z = train_ts.z.iloc[j]
                break
        if ind:
            train_table.append([x, y, z, train_l.label.iloc[i]])
    return train_table


def test_data(test_ts, test_l):
    """Grabbing the test data imported from files, and forming it into table that I can work with."""
    test_table = []
    for i in range(len(test_l)):
        x = 0
        y = 0
        z = 0
        time_stamp = test_l.timestamp.iloc[i]
        utc_time = test_l["UTC time"].iloc[i]
        ind = None
        for j in range(len(test_ts)):
            ind = (
                test_l.orig_index.iloc[i]
                if test_ts.timestamp.iloc[j] == test_l.timestamp.iloc[i]
                else None
            )
            if ind:
                x = test_ts.x.iloc[j]
                y = test_ts.y.iloc[j]
                z = test_ts.z.iloc[j]
                break
        if ind:
            test_table.append([ind, time_stamp, utc_time, x, y, z, 0, 0])
    return test_table


def sort_results(test, predict, prob):
    """Grabbing the logistic predictions and probabilities, and forming them into a results table."""
    results = []
    for i in range(len(test)):
        results.append(
            [
                test.orig_index.iloc[i],
                test.timestamp.iloc[i],
                test["UTC time"].iloc[i],
                predict[i],
                prob[i],
            ]
        )
    return results


def sift_training_by_classification(train):
    """The training table has over 50% of classification 2, so this function counts the number of entries for each classification, grabs the minimum count, and then selects that many entries from each category. Without making the counts even I can only get results of all classification 2."""
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    for i in range(len(train)):
        t1 += 1 if train.label.iloc[i] == 1 else 0
        t2 += 1 if train.label.iloc[i] == 2 else 0
        t3 += 1 if train.label.iloc[i] == 3 else 0
        t4 += 1 if train.label.iloc[i] == 4 else 0
    min_count = min(t1, t2, t3, t4)
    print("min count: ", min_count)
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    train_sift = []
    for j in range(len(train)):
        label = train.label.iloc[j]
        if label == 1 and s1 <= min_count:
            train_sift.append([train.x.iloc[j], train.y.iloc[j], train.z.iloc[j], train.label.iloc[j]])
            s1 += 1
        if label == 2 and s2 <= min_count:
            train_sift.append([train.x.iloc[j], train.y.iloc[j], train.z.iloc[j], train.label.iloc[j]])
            s2 += 1
        if label == 3 and s3 <= min_count:
            train_sift.append([train.x.iloc[j], train.y.iloc[j], train.z.iloc[j], train.label.iloc[j]])
            s3 += 1
        if label == 4 and s4 <= min_count:
            train_sift.append([train.x.iloc[j], train.y.iloc[j], train.z.iloc[j], train.label.iloc[j]])
            s4 += 1
    return train_sift


# sifting data
train_table = pd.DataFrame(
    train_data(df_train_time_series, df_train_labels), columns=["x", "y", "z", "label"]
)

# Using the full training data set gives me results of all classification 2. So, I am trying different sifting methods to try and create better results.

# creating a table with an equal number of classifications. This gives me results that include a mix of all categories, but the teacher says they are wrong.
train_table = pd.DataFrame(sift_training_by_classification(train_table), columns=["x", "y", "z", "label"])

#creating a table with a random selection of entries, with a quantity that is equal to the testing data. This results in all classification 2.
# train_table = train_table.sample(n=125)

#Using the base table, without modification. This results in all classification 2.
# train_table = train_table


test_table = pd.DataFrame(
    test_data(df_test_time_series, df_test_labels),
    columns=["orig_index", "timestamp", "UTC time", "x", "y", "z", "label", "accuracy"],
)

# printing tables to colsole for testing purposes
print("train table", train_table)
print("test table", test_table)

# forming tables to push through logistic regression
train_X = []
train_y = []
for i in range(len(train_table)):
    train_X.append(
        [train_table.x.iloc[i], train_table.y.iloc[i], train_table.z.iloc[i]]
    )
    train_y.append(train_table.label.iloc[i])

test_X = []
for k in range(len(test_table)):
    test_X.append([test_table.x.iloc[k], test_table.y.iloc[k], test_table.z.iloc[k]])

# Trying with and without normalize
clf = LogisticRegression().fit(normalize(train_X), train_y)
# clf = LogisticRegression().fit(train_X, train_y)

predict = clf.predict(test_X[:])
prob = clf.predict_proba(test_X[:])

results = pd.DataFrame(
    sort_results(test_table, predict, prob),
    columns=["", "timestamp", "UTC time", "label", "accuracy"],
)

# record results to file
results.to_csv(os.path.join(data_dir, "test_labels.csv"), sep=",", index=False)

# end_process timer
end_time = datetime.datetime.now()

# print elasped time to console
print("Run time: " + str(end_time - start_time)[5:] + " s.ms")

# print results to console. printing for testing purposes
label_list = ""
for i in range(len(results)):
    label_list = label_list + str(results.label.iloc[i]) + ","

print(label_list)
