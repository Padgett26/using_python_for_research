# Needed imports
import numpy as np
import pandas as pd
import datetime
import os
from math import sqrt

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
    train_table = []
    for i in range(len(train_l)):
        x = 0
        y = 0
        z = 0
        ind = None
        for j in range(len(train_ts)):
            ind = train_l.orig_index.iloc[i] if train_ts.timestamp.iloc[j] == train_l.timestamp.iloc[i] else None
            if ind:
                x = train_ts.x.iloc[j]
                y = train_ts.y.iloc[j]
                z = train_ts.z.iloc[j]
                break
        if ind:
            train_table.append([x, y, z, train_l.label.iloc[i]])
    return train_table


def test_data(test_ts, test_l):
    test_table = []
    for i in range(len(test_l)):
        x = 0
        y = 0
        z = 0
        time_stamp = test_l.timestamp.iloc[i]
        utc_time = test_l["UTC time"].iloc[i]
        ind = None
        for j in range(len(test_ts)):
            ind = test_l.orig_index.iloc[i] if test_ts.timestamp.iloc[j] == test_l.timestamp.iloc[i] else None
            if ind:
                x = test_ts.x.iloc[j]
                y = test_ts.y.iloc[j]
                z = test_ts.z.iloc[j]
                break
        if ind:
            test_table.append([ind, time_stamp, utc_time, x, y, z, 0, 0])
    return test_table


def training(train_table):
    training_stats = []
    for t in range(1, 5):
        x = []
        y = []
        z = []
        for r in range(len(train_table)):
            if train_table.label.iloc[r] == t:
                x.append(train_table.x.iloc[r])
                y.append(train_table.y.iloc[r])
                z.append(train_table.z.iloc[r])
        training_stats.append([np.mean(x), np.mean(y), np.mean(z), t])
    return training_stats


def find_accuracy(d):
    total = sum(d)
    min_d = np.min(d)
    accuracy = (1 - (min_d / total))
    return accuracy


def find_label(x, y, z, ts):
    d = []  # a list of distances to each training_stats points
    for i in range(4):
        d.append(sqrt(((ts[i][0] - x)**2 + (ts[i][1] - y)**2) + (ts[i][2] - z)**2))
    label = 0
    t = 1
    x = np.min(d)
    for i in range(len(d)):
        if d[i] == x:
            label = (t)
            break
        t += 1
    accuracy = find_accuracy(d)
    return (label, accuracy)


def check_neighbors(train_table, test_table, distance):
    results = []
    for i in range(len(test_table)):
        X_1 = test_table.x.iloc[i]
        Y_1 = test_table.y.iloc[i]
        Z_1 = test_table.z.iloc[i]
        counts = []
        for j in range(len(train_table)):
            x = train_table.x.iloc[j]
            y = train_table.y.iloc[j]
            z = train_table.z.iloc[j]
            label = train_table.label.iloc[j]
            if (sqrt(((X_1 - x)**2 + (Y_1 - y)**2) + (Z_1 - z)**2)) < distance:
                counts.append(label)
        l1 = 0
        l2 = 0
        l3 = 0
        l4 = 0
        for num in range(len(counts)):
            l1 += 1 if counts[num] == 1 else 0
            l2 += 1 if counts[num] == 2 else 0
            l3 += 1 if counts[num] == 3 else 0
            l4 += 1 if counts[num] == 4 else 0

        m = max(l1, l2, l3, l4)
        total = l1 + l2 + l3 + l4
        if total == 0:
            print("total: ", total)
        if l1 == m:
            found_label = 1
        elif l2 == m:
            found_label = 2
        elif l3 == m:
            found_label = 3
        elif l4 == m:
            found_label = 4
        else:
            found_label = 0
        a = (m / total) if total > 0 else 0
        results.append([test_table.orig_index.iloc[i], test_table.timestamp.iloc[i], test_table["UTC time"].iloc[i], found_label, a])
    return results


def testing(train_table, test_table):
    training_stats = training(train_table)
    results = []
    for i in range(len(test_table)):
        (label, accuracy) = find_label(test_table.x.iloc[i], test_table.y.iloc[i], test_table.z.iloc[i], training_stats)
        results.append([test_table.orig_index.iloc[i], test_table.timestamp.iloc[i], test_table["UTC time"].iloc[i], label, accuracy])
    return results


# sifting data
train_table = pd.DataFrame(train_data(df_train_time_series, df_train_labels), columns=['x', 'y', 'z', 'label'])

test_table = pd.DataFrame(test_data(df_test_time_series, df_test_labels), columns=['orig_index', 'timestamp', 'UTC time', 'x', 'y', 'z', 'label', 'accuracy'])

# results = pd.DataFrame(testing(train_table, test_table), columns=["", "timestamp", "UTC time", "label", "accuracy"])

results = pd.DataFrame(check_neighbors(train_table, test_table, 0.68), columns=["", "timestamp", "UTC time", "label", "accuracy"])

results.to_csv(os.path.join(data_dir, 'test_labels.csv'), sep=',', index=False)

# end_process timer
end_time = datetime.datetime.now()

# print elasped time to console
print("Run time: " + str(end_time - start_time)[5:] + " s.ms")

label_list = ""
for i in range(len(results)):
    label_list = label_list + str(results.label.iloc[i]) + ","

print(label_list)

