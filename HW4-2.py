import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

data_dir = "./data"

birddata = pd.read_csv(os.path.join(data_dir, "bird_tracking.csv"), index_col=0)
# print(birddata.head())

# First, use `groupby()` to group the data by "bird_name".
bird_names = np.unique(birddata.bird_name)

# mean_speeds = {}
# mean_altitudes = {}
# for name in bird_names:
#     mean_speeds[name] =  np.mean(birddata.speed_2d[birddata.bird_name == name])
#     mean_altitudes[name] = np.mean(birddata.altitude[birddata.bird_name == name])

# print("Mean speeds", mean_speeds)
# print("Mean altitudes", mean_altitudes)

# print (birddata[: 10])
# Use `groupby()` to group the data by date.
# mean_altitudes_perday = {}
# for g in data_dates:
#     date = datetime.datetime.strftime(g, "%Y-%m-%d")
#     mean_altitudes_perday[date] = np.mean(birddata.altitude[birddata.date == g])

# def bird_altitude_mean(bird, date):
#     alts = []
#     for i in range(len(birddata)):
#         if birddata.loc[i, "bird_name"] == bird and birddata.loc[i, "date"] == date:
#             alts.append(birddata.loc[i, "altitude"])
#     print(np.mean(alts))

# bird_altitude_mean("Eric", "2013-08-18")

def birds_speed_day(bird, day):
    speeds = {}
    ind = []
    for l in range(len(birddata)):
        if birddata.bird_name.iloc[l] == bird and birddata.date.iloc[l] == day:
            ind.append(l)
    speeds[day] = np.mean(birddata.speed_2d[ind])
    return speeds

def birds_speed(bird):
    speeds = {}
    days = np.unique(birddata.date)
    for day in days:
        ind = []
        for l in range(len(birddata)):
            if birddata.bird_name.iloc[l] == bird and birddata.date.iloc[l] == day:
                ind.append(l)
        speeds[day] = np.mean(birddata.speed_2d[ind])
    return speeds

print(birds_speed_day("Nico", "2014-04-04"))
# eric_daily_speed  = pd.Series(birds_speed("Eric"))
# sanne_daily_speed = pd.Series(birds_speed("Sanne"))
# nico_daily_speed  = pd.Series(birds_speed("Nico"))

# eric_daily_speed.plot(label="Eric")
# sanne_daily_speed.plot(label="Sanne")
# nico_daily_speed.plot(label="Nico")
# plt.legend(loc="upper left")
# plt.show()

