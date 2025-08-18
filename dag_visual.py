import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def _coerce_edge_weights(adj: np.ndarray, edge_probs):
    """
    Turn various edge_probs inputs into a dict {(i,j): weight}.
    Accepted forms:
      - None -> all existing edges weight=1.0
      - 2D array -> same shape as adj, use edge_probs[i,j] where adj[i,j]>0
      - 1D array -> length n-1, interpreted as parents -> last node (useful for 3-node toy)
    """
    n = adj.shape[0]
    weights = {}

    if edge_probs is None:
        # default to 1.0 on every present edge
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    weights[(i, j)] = 1.0
        return weights

    edge_probs = np.asarray(edge_probs)

    if edge_probs.ndim == 2:
        assert edge_probs.shape == adj.shape, "edge_probs matrix must match adj shape"
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    weights[(i, j)] = float(edge_probs[i, j])
        return weights

    if edge_probs.ndim == 1:
        # Convenience: treat as probabilities of edges into the last node
        assert len(edge_probs) == n - 1, "1D edge_probs must have length n-1 (parents -> last node)"
        last = n - 1
        for i in range(n - 1):
            if adj[i, last] > 0:
                weights[(i, last)] = float(edge_probs[i])
        # Any other edges get weight=1
        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0 and (i, j) not in weights:
                    weights[(i, j)] = 1.0
        return weights

    raise ValueError("Unsupported edge_probs format.")


def plot_dag(adj, names=None, edge_probs=None, layout: str = "shell", ax=None):
    """
    Plot a directed acyclic graph from an adjacency matrix.

    Args:
        adj (array-like, n x n): adjacency (parents -> child). Nonzero = edge present.
        names (list[str] | None): node labels; defaults to X0, X1, ...
        edge_probs (None | 1D | 2D array): optional edge strengths/probabilities.
            - None: all edges drawn with equal color
            - 2D: same shape as adj, uses values where adj>0
            - 1D: length n-1, interpreted as parents -> last node only
        layout (str): "shell" (default), "spring", or "kamada_kawai".
        ax (matplotlib.axes.Axes | None): axes to draw on. Creates one if None.

    Returns:
        matplotlib.axes.Axes: the axes with the drawn graph.
    """
    adj = np.asarray(adj)
    n = adj.shape[0]
    labels = names if names is not None else [f"X{i}" for i in range(n)]

    # Build graph
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, label=labels[i])

    weights = _coerce_edge_weights(adj, edge_probs)
    for (i, j), w in weights.items():
        G.add_edge(i, j, weight=float(w))

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.shell_layout(G)

    # Prepare drawing primitives
    node_labels = {i: G.nodes[i]["label"] for i in G.nodes}
    edge_wts = [G[u][v]["weight"] for u, v in G.edges] if G.number_of_edges() > 0 else []

    # Create axes if needed
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        created_ax = True

    # Draw nodes/edges
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        labels=node_labels,
        node_size=1500,
        node_color="#e0f3f8",
        arrows=True,
        arrowsize=20,
        edge_color=edge_wts if edge_wts else "#888888",
        edge_cmap=plt.cm.viridis if edge_wts else None,
        width=2,
    )

    # Colorbar only if we actually have weighted edges and not all weights equal
    if edge_wts:
        vmin, vmax = float(np.min(edge_wts)), float(np.max(edge_wts))
        if vmax - vmin < 1e-12:
            # avoid a degenerate colorbar; skip when all weights equal
            pass
        else:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Edge weight / P(edge)")

    ax.set_title("Current DAG")
    ax.axis("off")

    if created_ax:
        plt.tight_layout()
        plt.show()

    return ax