import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli as brn
import numpy as np
from collections import Counter
import os
import pandas as pd

data_dir = "./data"

df  = pd.read_csv(os.path.join(data_dir, "individual_characteristics.csv"), low_memory=False, index_col=0)
df1 = df.query("village == 1")
df2 = df.query("village == 2")

data_filepath1 = "key_vilno_1.csv"
data_filepath2 = "key_vilno_2.csv"
# pid1 = pd.read_csv(data_filepath1, low_memory=False, index_col=0)
# pid2 = pd.read_csv(data_filepath2, low_memory=False, index_col=0)

A1 = np.array(pd.read_csv(os.path.join(data_dir, "adj_allVillageRelationships_vilno1.csv"), index_col=0))
A2 = np.array(pd.read_csv(os.path.join(data_dir, "adj_allVillageRelationships_vilno2.csv"), index_col=0))
G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

pid1 = pd.read_csv(os.path.join(data_dir, data_filepath1), dtype=int)['0'].to_dict()
pid2 = pd.read_csv(os.path.join(data_dir, data_filepath2), dtype=int)['0'].to_dict()


sex1 = {}
caste1 = {}
religion1 = {}
for row_index, row in df1.iterrows():
    sex1[row['pid']] = row['resp_gend']
    caste1[row['pid']] = row['caste']
    religion1[row['pid']] = row['religion']

sex2 = {}
caste2 = {}
religion2 = {}
for row_index, row in df2.iterrows():
    sex2[row['pid']] = row['resp_gend']
    caste2[row['pid']] = row['caste']
    religion2[row['pid']] = row['religion']


def marginal_prob(chars):
    frequencies = dict(Counter(chars.values()))
    sum_frequencies = sum(frequencies.values())
    return {char: freq / sum_frequencies for char, freq in frequencies.items()}


def chance_homophily(chars):
    marginal_probs = marginal_prob(chars)
    return np.sum(np.square(list(marginal_probs.values())))


# print("Village 1 sex:", chance_homophily(sex1))
# print("Village 1 caste:", chance_homophily(caste1))
# print("Village 1 religion:", chance_homophily(religion1))
# print("Village 2 sex:", chance_homophily(sex2))
# print("Village 2 caste:", chance_homophily(caste2))
# print("Village 2 religion:", chance_homophily(religion2))
# favorite_colors = {
#     "ankit":  "red",
#     "xiaoyu": "blue",
#     "mary":   "blue"
# }

# m_color_homophily =marginal_prob(favorite_colors)
# print(m_color_homophily)

# color_homophily = chance_homophily(favorite_colors)
# print(color_homophilyl)

def homophily(G, chars, IDs):
    """
    Given a network G, a dict of characteristics chars for node IDs,
    and dict of node IDs for each node in the network,
    find the homophily of the network.
    """
    num_same_ties = 0
    num_ties = 0
    for n1, n2 in G.edges():
        if IDs[n1] in chars and IDs[n2] in chars:
            if G.has_edge(n1, n2):
                num_ties += 1
                if chars[IDs[n1]] == chars[IDs[n2]]:
                    num_same_ties += 1
    return (num_same_ties / num_ties)

print("V1 sex observ: ", homophily(G1, sex1, pid1))
print("V1 sex chance: ", chance_homophily(sex1))
print("V1 caste observ: ", homophily(G1, caste1, pid1))
print("V1 caste chance: ", chance_homophily(caste1))
print("V1 religion observ: ", homophily(G1, religion1, pid1))
print("V1 religion chance: ", chance_homophily(religion1))
print("V2 sex observ: ", homophily(G2, sex2, pid2))
print("V2 sex chance: ", chance_homophily(sex2))
print("V2 caste observ: ", homophily(G2, caste2, pid2))
print("V2 caste chance: ", chance_homophily(caste2))
print("V2 religion observ: ", homophily(G2, religion2, pid2))
print("V2 religion chance: ", chance_homophily(religion2))

