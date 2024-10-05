import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering

data_dir = "./data"

whiskey = pd.read_csv(os.path.join(data_dir, "whiskies.txt"))
whiskey["Region"] = pd.read_csv(os.path.join(data_dir, "regions.txt"))

flavors = whiskey.iloc[:, 2:14]

# print(flavors)

corr_flavors = pd.DataFrame.corr(flavors)

# print(corr_flavors)

plt.figure(figsize=(10, 10))
plt.pcolor(corr_flavors)
plt.colorbar()
# plt.savefig("images/corr_flavors.pdf")

corr_whiskey = pd.DataFrame.corr(flavors.transpose())

# print(corr_flavors)

plt.figure(figsize=(10, 10))
plt.pcolor(corr_whiskey)
plt.axis("tight")
plt.colorbar()
# plt.savefig("images/corr_whiskey.pdf")

model = SpectralCoclustering(n_clusters=6, random_state=0)
model.fit(corr_whiskey)

# print(np.sum(model.rows_, axis=1))
# print(model.row_labels_)

whiskey['Group'] = pd.Series(model.row_labels_, index=whiskey.index)
whiskey = whiskey.iloc[np.argsort(model.row_labels_)]
whiskey = whiskey.reset_index(drop=True)
correlations = pd.DataFrame.corr(whiskey.iloc[:, 2:14].transpose())
correlations = np.array(correlations)

plt.figure(figsize=(14, 7))
plt.subplot(121)
plt.pcolor(corr_whiskey)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")
plt.savefig("images/correlations.pdf")

