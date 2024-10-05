import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli as brn
import numpy as np

# G = nx.Graph()
# G.add_node(1)
# G.add_nodes_from([2, 3, "u", "v"])
# print(G.nodes())

# G.add_edge(1,2)
# G.add_edge("u", "v")
# G.add_edges_from([(1,3), (1,4), (1,5), (1,6)])
# print(G.number_of_nodes())
# print(G.number_of_edges())

# G = nx.karate_club_graph()
# nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
# plt.savefig("images/karate_club_graph.pdf")
# print(G.number_of_nodes())
# print(G.number_of_edges())
# print(G.degree(0) is G.degree()[0])

N = 100
p = 0.3

def er_graph(N, p):
    """create random graph using N nodes and p probability of edges"""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and brn.rvs(p=p):
                G.add_edge(node1, node2)
    return G

# nx.draw(er_graph(N, p), node_size=40, node_color="gray")
# plt.savefig("images/er1.pdf")

def plot_degree_distribution(G):
    degree_sequence = [d for n, d in G.degree()]
    plt.hist(degree_sequence, histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree distribution")

# for i in range(3)
#     plot_degree_distribution(er_graph(N, p))

# plt.savefig("images/hist_3.pdf")

V1 = np.loadtxt("data/adj_allVillageRelationships_vilno_1.csv", delimiter=",")
V2 = np.loadtxt("data/adj_allVillageRelationships_vilno_2.csv", delimiter=",")
G1 = nx.to_networkx_graph(V1)
G2 = nx.to_networkx_graph(V2)

def basic_net_stats(G):
    print("Number of nodes: %d" % G.number_of_nodes())
    print("Number of edges: %d" % G.number_of_edges())
    degree_sequence = [d for n, d in G.degree()]
    print("Average degree: %.2f" % np.mean(degree_sequence))

# basic_net_stats(G1)
# basic_net_stats(G2)

# plot_degree_distribution(G1)
# plot_degree_distribution(G2)

# plt.savefig("images/village_hist.pdf")

G1_LIST = [len(c) for c in sorted(nx.connected_components(G1), key=len, reverse=True)]
G2_LIST = [len(c) for c in sorted(nx.connected_components(G2), key=len, reverse=True)]

print(G1_LIST)
print(G2_LIST)

print(G1_LIST[0] / sum(G1_LIST))
print(G2_LIST[0] / sum(G2_LIST))

_pos1 = nx.spring_layout(G1)
_pos2 = nx.spring_layout(G2)
plt.figure()
nx.draw_networkx_edges(G1, _pos1, alpha=0.3, edge_color="k")
nx.draw_networkx_nodes(G1, _pos1, node_color="red", node_size=2)
plt.savefig("images/village1.pdf")

plt.figure()
nx.draw_networkx_edges(G2, _pos2, alpha=0.3, edge_color="k")
nx.draw_networkx_nodes(G2, _pos2, node_color="green", node_size=2)
plt.savefig("images/village2.pdf")
