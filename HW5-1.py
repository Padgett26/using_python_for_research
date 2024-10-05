import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os


data_dir = "./data"

df = pd.read_csv(os.path.join(data_dir, "movie_data.csv"), index_col=0)

ind = []
for i in range(len(df)):
    ind.append(1 if df["revenue"].iloc[i] > df["budget"].iloc[i] else 0)

prof = pd.DataFrame({"profitable": ind})

df = df.merge(prof, how='inner', on=df.index)
df.pop('key_0')

df = df.replace(np.inf, np.nan)
df = df.dropna()

prof_count = 0
tot_count = 0
for i in range(len(df)):
    prof_count += df['profitable'].iloc[i]
    tot_count += 1


# print("Profitable count: ", prof_count)
# print("Total count: ", tot_count)
# print(prof.tail())

genres = []
indiv_gens = []

for i in range(len(df)):
    gs = df['genres'].iloc[i]
    g = gs.split(',')
    for k in range(len(g)):
        g[k] = g[k].strip()

    for j in g:
        genres.append(j)
    indiv_gens.append({i: g})

genres = np.unique(genres)
# print(genres)

for genre in genres:
    gen_list = []
    for i in range(len(indiv_gens)):
        t = 0
        if genre in df['genres'].iloc[i]:
            t = 1
        gen_list.append(t)
    g = pd.DataFrame(gen_list, columns=[genre])
    df = df.merge(g, how='inner', on=df.index)
    df.pop('key_0')

# print(df.head())
# print("# of genres: ", len(genres))

regression_target = "revenue"
classification_target = "profitable"

continuous_covariates = ['budget', 'popularity', 'runtime', 'vote_count', 'vote_average']
outcomes_and_continuous_covariates = continuous_covariates + [regression_target, classification_target]
plotting_variables = ['budget', 'popularity', regression_target]

axes = pd.plotting.scatter_matrix(df[plotting_variables], alpha=0.15, color=(0,0,0), hist_kwds={"color":(0,0,0)}, facecolor=(1,0,0))
# plt.show()

# determine the skew
budget =  np.log10(1 + df['budget']).skew()
popularity =  np.log10(1 + df['popularity']).skew()
runtime =  np.log10(1 + df['runtime']).skew()
vote_count =  np.log10(1 + df['vote_count']).skew()
vote_average =  np.log10(1 + df['vote_average']).skew()
revenue =  np.log10(1 + df['revenue']).skew()
profitable =  np.log10(1 + df['profitable']).skew()

print("budget: ", budget)
print("popularity", popularity)
print("runtime", runtime)
print("vote_count", vote_count)
print("vote_average", vote_average)
print("revenue", revenue)
print("profitable", profitable)

df.to_csv("data/movies_clean.csv")

