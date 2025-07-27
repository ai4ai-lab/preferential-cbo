import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_dag(adj, names=None, edge_probs=None):
    G = nx.DiGraph()
    n = adj.shape[0]
    for i in range(n):
        G.add_node(i, label=names[i] if names else f"X{i}")
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                w = edge_probs[i, j] if edge_probs is not None else 1.0
                G.add_edge(i, j, weight=w)

    pos = nx.shell_layout(G)  # or spring_layout
    edge_colors = [G[u][v]['weight'] for u,v in G.edges()]
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    nx.draw(G, pos,
            with_labels=True, labels=labels,
            node_size=1500, node_color="#e0f3f8",
            arrowsize=20, edge_color=edge_colors,
            edge_cmap=plt.cm.viridis, width=2)
    if edge_probs is not None:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                   norm=plt.Normalize(vmin=np.min(edge_colors), vmax=np.max(edge_colors)))
        plt.colorbar(sm, label='P(edge)')
    plt.title("Current DAG / Parent structure")
    plt.show()