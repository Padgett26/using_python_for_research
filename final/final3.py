import datetime
import os

import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier

data_dir = "./data"

# start process timer
start_time = datetime.datetime.now()

train_covariates = pd.read_csv(os.path.join(data_dir, "train_time_series.csv"), index_col="timestamp").drop(columns=["Unnamed: 0", "UTC time", "accuracy"])
train_outcomes = pd.read_csv(os.path.join(data_dir, "train_labels.csv"), index_col="timestamp").drop(columns=["Unnamed: 0", "UTC time"])
train_table = pd.concat([train_covariates,train_outcomes], axis=1).dropna(axis=0, how="any")

X, y = train_table.iloc[:, 0:3], train_table.iloc[:, 3]

param_dist = {'n_estimators': randint(100, 375),
              'max_depth': randint(5, 20)}

rf = RandomForestClassifier()

rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, n_iter=10, cv=5)

rand_search.fit(X, y)

best_rf = rand_search.best_estimator_

# print to console the best parameters found in the collection of trees created. For reference.
print('Best hyperparameters:',  rand_search.best_params_)

to_pred_outcomes = pd.read_csv(os.path.join(data_dir, "test_labels_input.csv"))
list_of_timestamps = to_pred_outcomes["timestamp"]
to_pred_covariates = pd.read_csv(os.path.join(data_dir, "test_time_series.csv"), index_col="timestamp").drop(columns=["Unnamed: 0", "UTC time", "accuracy"])
to_pred_covariates = to_pred_covariates[to_pred_covariates.index.isin(list_of_timestamps)]

predictions = pd.Series(best_rf.predict(to_pred_covariates))
predictions.to_csv(os.path.join(data_dir, "test_labels.csv"))

end_time = datetime.datetime.now()

# print elasped time to console
print("Run time: " + str(end_time - start_time)[5:] + " s.ms")

