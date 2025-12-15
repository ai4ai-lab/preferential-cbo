import torch
import os
import numpy as np
from torch.utils.data import Dataset


def _make_rng(seed: int):
    """Reproducible random number generator with the given seed"""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


class PCBO_Dataset_Three(Dataset):
    """
    Simple 3-node dataset: X1 -> Y <- X2.
    Pairwise preferences (k=2). Utility = -|Y|.
    Outcome vector is [X1, X2, Y].
    """
    def __init__(self, n_queries=100, noise_std=0.1, domain=(-2.0, 2.0), seed=42):
        super().__init__()
        self.n_nodes = 3
        self.n_queries = n_queries
        self.noise_std = float(noise_std)
        self.domain = domain
        self.seed = seed
        self.rng = _make_rng(self.seed)

        # Names and adjacency (parents -> child)
        self.node_names = ["X1", "X2", "Y"]
        self.adj = torch.tensor(
            [
                [0, 0, 1],  # X1 -> Y
                [0, 0, 1],  # X2 -> Y
                [0, 0, 0],  # Y
            ],
            dtype=torch.int64,
        )

        # Generate pairwise queries
        self.queries = self._generate_pairwise_data(self.n_queries)

    # -------- SCM + Utility --------
    def _scm_y(self, x1, x2):
        """Nonlinear SCM for Y: Y = tanh(x1) + 0.5 * sin(x2) + eps."""
        eps = self._randn_scalar(self.noise_std)
        return torch.tanh(x1) + 0.5 * torch.sin(x2) + eps
    
    def _compute_intervention_outcome(self, node_idx, value):
        """
        Compute outcome of intervention do(X_i = value)
        Returns the full state after intervention
        """
        # Defaults as tensors
        x1 = self._randn_scalar(0.5)
        x2 = self._randn_scalar(0.5)
        y  = torch.tensor(0.0)

        if node_idx == 0:  # X1
            x1 = torch.as_tensor(value)  # Value can be float or tensor; this makes it tensor
        elif node_idx == 1:  # X2
            x2 = torch.as_tensor(value)
        elif node_idx == 2:  # Y (hard intervention)
            y  = torch.as_tensor(value)

        if node_idx != 2:
            y = self._scm_y(x1, x2)

        return torch.stack([x1, x2, y])
    
    def _true_utility(self, state, intervention_node=None):
        """
        True utility function (unknown to the algorithm)
        This is what we're trying to learn through preferences
        """
        # Utility depends on outcome Y and intervention cost
        utility = -torch.abs(state[2])  # Want Y close to 0
        if intervention_node is not None:
            utility -= 0.1  # Small cost for intervening
        return utility
    
    # -------- Data generation --------
    def _sample_intervention(self):
        node = int(torch.randint(0, self.n_nodes, (1,), generator=self.rng))
        val  = self._uniform_scalar(*self.domain).item()  # Store as float
        return node, val

    def _generate_pairwise_data(self, n_queries):
        qs = []
        for _ in range(n_queries):
            intervs, outs, utils = [], [], []

            for _ in range(2):  # pairwise
                node, val = self._sample_intervention()
                out = self._compute_intervention_outcome(node, val)
                util = self._true_utility(out, node)
                intervs.append((node, val))
                outs.append(out)
                utils.append(util)

            outs = torch.stack(outs)  # (2, 3)
            utils = torch.stack(utils).squeeze(-1)  # (2,)
            winner_idx = int(torch.argmax(utils))

            qs.append(
                {
                    "interventions": intervs,  # list[ (node, value), (node, value) ]
                    "outcomes": outs,  # (2, 3) [X1, X2, Y]
                    "utilities": utils,  # (2,)
                    "winner_idx": winner_idx,  # 0 or 1
                }
            )
        return qs
    
    # -------- Helper methods --------
    def _uniform_scalar(self, low, high):
        # 0-D tensor in [low, high)
        return low + (high - low) * torch.rand((), generator=self.rng)

    def _randn_scalar(self, std=1.0):
        # 0-D tensor ~ N(0, std^2)
        return std * torch.randn((), generator=self.rng)
    
    # -------- Public API --------
    def get_causal_graph(self):
        return self.adj.clone(), list(self.node_names)

    def __len__(self):
        return self.n_queries

    def __getitem__(self, idx):
        return self.queries[idx]
    

class PCBO_Dataset_Six(Dataset):
    """
    Harder 6-node dataset with mediator and nonlinear Y.
    Nodes: X1, X2, X3, X4 (mediator), X5, Y
    Edges: X1->X4, X3->X4; and X1,X2,X3,X4 -> Y. (X5 independent)
    Pairwise preferences (k=2). Utility = -|Y|.
    Outcome vector is [X1, X2, X3, X4, X5, Y].
    """
    def __init__(self, n_queries=100, noise_std=0.1, domain=(-2.0, 2.0), seed=42,
                 allow_intervene_on_mediator=True, allow_intervene_on_y=False):
        super().__init__()
        self.n_nodes = 6
        self.n_queries = n_queries
        self.noise_std = float(noise_std)
        self.domain = tuple(map(float, domain))
        self.seed = int(seed)
        self.rng = _make_rng(self.seed)
        self.allow_intervene_on_mediator = bool(allow_intervene_on_mediator)
        self.allow_intervene_on_y = bool(allow_intervene_on_y)

        self.node_names = ["X1", "X2", "X3", "X4", "X5", "Y"]

        # Adjacency (parents -> child), 6x6 matrix
        A = torch.zeros(6, 6, dtype=torch.int64)
        # X4 <- X1, X3
        A[0, 3] = 1
        A[2, 3] = 1
        # Y <- X1, X2, X3, X4
        A[0, 5] = 1
        A[1, 5] = 1
        A[2, 5] = 1
        A[3, 5] = 1
        self.adj = A

        self.queries = self._generate_pairwise_data(self.n_queries)

    # -------- SCM + Utility --------
    def _scm_forward(self, x1, x2, x3, x4):
        """
        x4 = 0.5*x1 + 0.5*x3 + η4
        y  = tanh(1.2*x1) + 0.9*x2^2 - 1.1*relu(x3) + 0.7*sin(0.8*x4) + 0.4*x1*x3 + ε
        """
        eta4 = torch.randn((), generator=self.rng) * self.noise_std
        x4_val = 0.5 * x1 + 0.5 * x3 + eta4

        eps = torch.randn((), generator=self.rng) * self.noise_std
        y = (
            1.0 * torch.tanh(1.2 * x1)
            + 0.9 * (x2 ** 2)
            - 1.1 * torch.relu(x3)
            + 0.7 * torch.sin(0.8 * x4_val)
            + 0.4 * (x1 * x3)
            + eps
        )
        return x4_val, y

    def _compute_intervention_outcome(self, node_idx, value):
        """
        do(X_node_idx = value). Other non-intervened features are sampled.
        Returns state: [X1, X2, X3, X4, X5, Y].
        """
        # Exogenous draws as 0-D tensors (reproducible)
        x1 = self._randn_scalar(1.0)
        x2 = self._randn_scalar(1.0)
        x3 = self._randn_scalar(1.0)
        x4 = torch.tensor(0.0)  # Set by SCM unless directly intervened
        x5 = self._randn_scalar(1.0)
        y  = torch.tensor(0.0)

        v = torch.as_tensor(value)

        # Apply intervention
        if node_idx == 0: x1 = v
        elif node_idx == 1: x2 = v
        elif node_idx == 2: x3 = v
        elif node_idx == 3 and self.allow_intervene_on_mediator: x4 = v
        elif node_idx == 4: x5 = v
        elif node_idx == 5 and self.allow_intervene_on_y: y = v

        # Roll forward
        if node_idx == 3 and self.allow_intervene_on_mediator:
            # x4 fixed by intervention; compute y with this x4
            eps = self._randn_scalar(self.noise_std)
            y = (
                1.0 * torch.tanh(1.2 * x1)
                + 0.9 * (x2 ** 2)
                - 1.1 * torch.relu(x3)
                + 0.7 * torch.sin(0.8 * x4)
                + 0.4 * (x1 * x3)
                + eps
            )
        elif not (node_idx == 5 and self.allow_intervene_on_y):
            # Usual structural propagation
            x4, y = self._scm_forward(x1, x2, x3, x4)

        return torch.stack([x1, x2, x3, x4, x5, y])
    
    def _true_utility(self, outcome, node_idx: int | None):
        """Maximize -|Y|; small intervention cost."""
        util = -outcome[5].abs()
        if node_idx is not None:
            util = util - 0.1
        return util
    
    # -------- Data generation --------
    def _allowed_nodes(self):
        # Start with X1,X2,X3,X5 (0,1,2,4)
        nodes = [0, 1, 2, 4]
        if self.allow_intervene_on_mediator:
            nodes.append(3)
        if self.allow_intervene_on_y:
            nodes.append(5)
        return nodes
    
    def _sample_intervention(self):
        allowed = self._allowed_nodes()
        idx = int(torch.randint(0, len(allowed), (1,), generator=self.rng))
        node = allowed[idx]
        val = self._uniform_scalar(*self.domain).item()  # Store as float for logs/printing
        return node, val

    def _generate_pairwise_data(self, n_queries):
        qs = []
        for _ in range(n_queries):
            intervs, outs, utils = [], [], []

            for _ in range(2):  # Pairwise
                node, val = self._sample_intervention()
                out = self._compute_intervention_outcome(node, val)
                util = self._true_utility(out, node)
                intervs.append((node, val))
                outs.append(out)
                utils.append(util)

            outs = torch.stack(outs)  # (2, 6)
            utils = torch.stack(utils).squeeze(-1)  # (2,)
            winner_idx = int(torch.argmax(utils))

            qs.append(
                {
                    "interventions": intervs,  # list[ (node, value), (node, value) ]
                    "outcomes": outs,  # (2, 6)
                    "utilities": utils,  # (2,)
                    "winner_idx": winner_idx,  # 0 or 1
                }
            )
        return qs
    
    # -------- Helpers --------
    def _uniform_scalar(self, low, high):
        # 0-D tensor in [low, high)
        return low + (high - low) * torch.rand((), generator=self.rng)

    def _randn_scalar(self, std=1.0):
        # 0-D tensor ~ N(0, std^2)
        return std * torch.randn((), generator=self.rng)

    # -------- Public API --------
    def get_causal_graph(self):
        return self.adj.clone(), list(self.node_names)

    def __len__(self):
        return self.n_queries

    def __getitem__(self, idx):
        return self.queries[idx]
    

def generate_er_dag(n, p, rng=None):
    """
    Generate an Erdős-Rényi DAG by sampling a random topological order and
    adding edges i->j (i<j in the order) with probability p.
    Returns adjacency (n x n) with 0 diag, strictly upper-triangular in the order.
    """
    rng = np.random.default_rng(None if rng is None else rng)
    order = rng.permutation(n)
    A = np.zeros((n, n), dtype=np.int64)
    for ii in range(n):
        for jj in range(ii + 1, n):
            if rng.random() < p:
                A[order[ii], order[jj]] = 1
    return A, order


def sample_linear_gaussian(A, n_samples=1000, w_min=0.5, w_max=1.5, noise=0.1, rng=None):
    """
    Simple linear-Gaussian SCM for simulation: X = W^T parents + eps.
    Returns data matrix X (n_samples x d) and weights W.
    """
    rng = np.random.default_rng(None if rng is None else rng)
    d = A.shape[0]
    W = rng.uniform(w_min, w_max, size=A.shape) * A
    X = np.zeros((n_samples, d), dtype=np.float32)
    order = list(np.argsort(np.sum(A, axis=0)))
    for t in range(n_samples):
        x = np.zeros(d, dtype=np.float32)
        for j in order:
            pa = np.where(A[:, j] == 1)[0]
            mean = (W[pa, j] @ x[pa]) if len(pa) else 0.0
            x[j] = mean + rng.normal(0, noise)
        X[t] = x
    return X, W


class PCBO_ERDataset(Dataset):
    """
    Scalable synthetic dataset backed by an ER DAG and linear-Gaussian SCM.
    Creates (context/features) -> latent utility u(x) = v^T x + eps for preference queries.
    """
    def __init__(self, n_nodes=15, edge_prob=0.15, n_queries=100, 
                 noise_std=0.1, domain=(-2.0, 2.0), seed=42):
        super().__init__()
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.n_queries = n_queries
        self.noise_std = float(noise_std)
        self.domain = tuple(map(float, domain))
        self.seed = seed
        self.rng = _make_rng(seed)
        
        # Generate DAG structure
        self.adj, self.order = generate_er_dag(n_nodes, edge_prob, rng=seed)
        
        # Generate SCM weights
        self.W = np.random.default_rng(seed).uniform(0.5, 1.5, size=self.adj.shape) * self.adj
        
        # Ground-truth linear utility for preference generation
        r = np.random.default_rng(seed).normal(0, 1, size=(n_nodes,))
        r = r / (np.linalg.norm(r) + 1e-8)
        self._r = torch.tensor(r, dtype=torch.float32)
        
        # Node names
        self.node_names = [f"X{k}" for k in range(self.n_nodes)]
        
        # Generate preference queries
        self.queries = self._generate_pairwise_data(self.n_queries)
    
    def _compute_intervention_outcome(self, node_idx, value):
        """
        Compute outcome of intervention do(X_i = value) using linear SCM.
        Returns the full state after intervention.
        """
        # Initialize with random exogenous noise
        outcome = np.random.randn(self.n_nodes) * self.noise_std
        
        # Set intervened node
        outcome[node_idx] = value
        
        # Forward propagate through SCM (respecting topological order)
        # Skip the intervened node
        order = list(np.argsort(np.sum(self.adj, axis=0)))
        
        for j in order:
            if j == node_idx:
                continue  # Skip intervened node
            
            parents = np.where(self.adj[:, j] == 1)[0]
            if len(parents) > 0:
                outcome[j] = np.sum(self.W[parents, j] * outcome[parents]) + \
                           np.random.randn() * self.noise_std
        
        return torch.tensor(outcome, dtype=torch.float32)
    
    def _true_utility(self, outcome, node_idx=None):
        """
        True utility function using linear model.
        """
        if not isinstance(outcome, torch.Tensor):
            outcome = torch.tensor(outcome, dtype=torch.float32)
        
        utility = torch.dot(self._r, outcome)
        
        # Add intervention cost
        if node_idx is not None:
            utility -= 0.1
            
        return utility
    
    def _sample_intervention(self):
        """Sample a random intervention."""
        node = int(torch.randint(0, self.n_nodes, (1,), generator=self.rng))
        val = float(self._uniform_scalar(*self.domain))
        return node, val
    
    def _uniform_scalar(self, low, high):
        return low + (high - low) * torch.rand((), generator=self.rng)
    
    def _generate_pairwise_data(self, n_queries):
        """Generate pairwise preference queries."""
        qs = []
        for _ in range(n_queries):
            intervs, outs, utils = [], [], []
            
            for _ in range(2):  # pairwise
                node, val = self._sample_intervention()
                out = self._compute_intervention_outcome(node, val)
                util = self._true_utility(out, node)
                intervs.append((node, val))
                outs.append(out)
                utils.append(util)
            
            outs = torch.stack(outs)
            utils = torch.stack(utils)
            winner_idx = int(torch.argmax(utils))
            
            qs.append({
                "interventions": intervs,
                "outcomes": outs,
                "utilities": utils,
                "winner_idx": winner_idx,
            })
        return qs
    
    def get_causal_graph(self):
        """Return adjacency matrix and node names."""
        return torch.tensor(self.adj, dtype=torch.int64), list(self.node_names)
    
    def __len__(self):
        return self.n_queries
    
    def __getitem__(self, idx):
        return self.queries[idx]
    

class PCBO_MedicalDataset(Dataset):
    """
    A small 'medical' DAG (treatment, biomarkers, risk factors, outcome).
    Nodes: Treat, Age, Smoke, BMI, BP, Biom1, Biom2, Out
    """
    def __init__(self, n_queries=100, noise_std=0.1, domain=(-2.0, 2.0), seed=42):
        super().__init__()
        self.n_nodes = 8
        self.n_queries = n_queries
        self.noise_std = float(noise_std)
        self.domain = tuple(map(float, domain))
        self.seed = seed
        self.rng = _make_rng(seed)
        
        self.node_names = ["Treat", "Age", "Smoke", "BMI", "BP", "Biom1", "Biom2", "Out"]
        
        # Build adjacency matrix
        A = np.zeros((8, 8), dtype=np.int64)
        # Risk factors -> biomarkers
        A[1, 4] = 1  # Age -> BP
        A[2, 4] = 1  # Smoke -> BP
        A[3, 4] = 1  # BMI -> BP
        A[4, 5] = 1  # BP -> Biom1
        A[3, 6] = 1  # BMI -> Biom2
        # Treatment -> biomarkers and outcome
        A[0, 5] = 1  # T -> Biom1
        A[0, 6] = 1  # T -> Biom2
        # Biomarkers & risk factors -> outcome
        A[5, 7] = 1; A[6, 7] = 1
        A[1, 7] = 1; A[3, 7] = 1; A[4, 7] = 1
        
        self.adj = A
        
        # Generate SCM weights
        self.W = np.random.default_rng(seed).uniform(0.5, 1.5, size=A.shape) * A
        
        # Ground-truth utility emphasizing Outcome (node 7) and Treatment (node 0)
        r = np.zeros((8,), dtype=np.float32)
        r[7] = 1.0  # Outcome
        r[0] = 0.3  # Treatment
        r = r / (np.linalg.norm(r) + 1e-8)
        self._r = torch.tensor(r, dtype=torch.float32)
        
        # Generate preference queries
        self.queries = self._generate_pairwise_data(self.n_queries)
    
    def _compute_intervention_outcome(self, node_idx, value):
        """
        Compute outcome of intervention do(X_i = value) using linear SCM.
        """
        # Initialize with random exogenous noise
        outcome = np.random.randn(self.n_nodes) * self.noise_std
        
        # Add some baseline values for realism
        outcome[1] += 50  # Age around 50
        outcome[3] += 25  # BMI around 25
        
        # Set intervened node
        outcome[node_idx] = value
        
        # Forward propagate through SCM
        order = list(np.argsort(np.sum(self.adj, axis=0)))
        
        for j in order:
            if j == node_idx:
                continue
            
            parents = np.where(self.adj[:, j] == 1)[0]
            if len(parents) > 0:
                outcome[j] = np.sum(self.W[parents, j] * outcome[parents]) + \
                           np.random.randn() * self.noise_std
        
        return torch.tensor(outcome, dtype=torch.float32)
    
    def _true_utility(self, outcome, node_idx=None):
        """
        Medical utility: maximize good outcome, minimize treatment.
        """
        if not isinstance(outcome, torch.Tensor):
            outcome = torch.tensor(outcome, dtype=torch.float32)
        
        utility = torch.dot(self._r, outcome)
        
        # Add intervention cost (especially for treatment)
        if node_idx is not None:
            if node_idx == 0:  # Treatment node
                utility -= 0.2
            else:
                utility -= 0.1
                
        return utility
    
    def _sample_intervention(self):
        """Sample a random intervention."""
        node = int(torch.randint(0, self.n_nodes, (1,), generator=self.rng))
        val = float(self._uniform_scalar(*self.domain))
        return node, val
    
    def _uniform_scalar(self, low, high):
        return low + (high - low) * torch.rand((), generator=self.rng)
    
    def _generate_pairwise_data(self, n_queries):
        """Generate pairwise preference queries."""
        qs = []
        for _ in range(n_queries):
            intervs, outs, utils = [], [], []
            
            for _ in range(2):
                node, val = self._sample_intervention()
                out = self._compute_intervention_outcome(node, val)
                util = self._true_utility(out, node)
                intervs.append((node, val))
                outs.append(out)
                utils.append(util)
            
            outs = torch.stack(outs)
            utils = torch.stack(utils)
            winner_idx = int(torch.argmax(utils))
            
            qs.append({
                "interventions": intervs,
                "outcomes": outs,
                "utilities": utils,
                "winner_idx": winner_idx,
            })
        return qs
    
    def get_causal_graph(self):
        """Return adjacency matrix and node names."""
        return torch.tensor(self.adj, dtype=torch.int64), list(self.node_names)
    
    def __len__(self):
        return self.n_queries
    
    def __getitem__(self, idx):
        return self.queries[idx]
    

# =========================================================================
# BIF / BNLearn datasets (alarm, asia, cancer, child, earthquake, diabetes)
# =========================================================================

try:
    from pgmpy.readwrite import BIFReader
except Exception:
    BIFReader = None


def _load_bif_model(bif_path: str):
    if BIFReader is None:
        raise ImportError("pgmpy is required for BIF datasets. Install with: pip install pgmpy")
    reader = BIFReader(bif_path)
    model = reader.get_model()
    return model


def _adj_from_model(model, node_order):
    """Binary adjacency A[i,j]=1 if node_order[i] -> node_order[j]."""
    idx = {n: i for i, n in enumerate(node_order)}
    d = len(node_order)
    A = np.zeros((d, d), dtype=np.int64)
    for u, v in model.edges():
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1
    np.fill_diagonal(A, 0)
    return A


def _topo_order_from_edges(nodes, edges):
    """Deterministic Kahn topo-sort without networkx dependency."""
    nodes = list(nodes)
    indeg = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}

    for u, v in edges:
        if u not in indeg or v not in indeg:
            continue
        children[u].append(v)
        indeg[v] += 1

    # deterministic: initial queue respects nodes order
    queue = [n for n in nodes if indeg[n] == 0]
    out = []

    while queue:
        n = queue.pop(0)
        out.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)

    # If something goes wrong (shouldn't for a BN), fallback to given order
    if len(out) != len(nodes):
        return nodes
    return out


def _sample_from_cpd_values(cpd_values: np.ndarray, parent_values: list[int] | None, rng: np.random.Generator) -> int:
    """
    cpd_values: ndarray shaped (card_child, card_pa1, card_pa2, ...)
    parent_values: list of int states in EXACT order of cpd.variables[1:].
    """
    arr = np.asarray(cpd_values, dtype=np.float64)

    if parent_values is None or len(parent_values) == 0:
        probs = arr.reshape(-1)  # (card_child,)
    else:
        slicer = (slice(None),) + tuple(int(v) for v in parent_values)
        probs = arr[slicer].reshape(-1)  # (card_child,)

    s = probs.sum()
    if s <= 0:
        # extremely defensive: fallback to uniform
        probs = np.ones_like(probs, dtype=np.float64) / len(probs)
    else:
        probs = probs / s

    return int(rng.choice(len(probs), p=probs))


class PCBO_BIFDataset(Dataset):
    """
    Generic wrapper for BNLearn-style .bif Bayesian networks.

    Files expected:
        data_dir/{name}.bif   e.g. data/alarm.bif

    Interventions:
        do(X_i = state_index), where state_index in [0, card_i-1].

    Outcomes:
        full assignment vector (d,) stored as float32 tensor
        (values are integer-coded states, but float32 for your flow).

    Preferences:
        pairwise (k=2). Winner chosen by utility.

    Utility:
        default: make target node hit `desired_state`:
            u = -abs(x[target]-desired_state) - intervention_cost
    """

    def __init__(
        self,
        name: str,
        data_dir: str = "data",
        n_queries: int = 100,
        seed: int = 42,
        intervention_cost: float = 0.1,
        target_node: str | None = None,  # if None -> last node in node_names
        desired_state: int = 0,
        allowed_intervention_nodes: list[str] | None = None,  # list of node names
    ):
        super().__init__()
        self.name = str(name)
        self.data_dir = str(data_dir)
        self.n_queries = int(n_queries)
        self.seed = int(seed)
        self.intervention_cost = float(intervention_cost)

        bif_path = os.path.join(self.data_dir, f"{self.name}.bif")
        if not os.path.exists(bif_path):
            raise FileNotFoundError(f"Could not find BIF file: {bif_path}")

        self._np_rng = np.random.default_rng(self.seed)
        self._torch_rng = torch.Generator().manual_seed(self.seed)

        # --- Load BN model ---
        self.model = _load_bif_model(bif_path)

        # Stable node order
        self.node_names = list(self.model.nodes())
        self.n_nodes = len(self.node_names)
        self.name_to_idx = {n: i for i, n in enumerate(self.node_names)}

        # Adjacency
        self.adj = torch.tensor(_adj_from_model(self.model, self.node_names), dtype=torch.int64)

        # CPDs map
        cpds = {cpd.variable: cpd for cpd in self.model.get_cpds()}
        missing = [n for n in self.node_names if n not in cpds]
        if missing:
            raise ValueError(f"Missing CPDs for nodes: {missing}")
        self.cpds = cpds

        # Cardinalities in node order (list indexed by node index)
        self.cardinalities = [int(self.model.get_cardinality(v)) for v in self.node_names]

        # Target setup
        if target_node is None:
            self.target_idx = self.n_nodes - 1
        else:
            if target_node not in self.name_to_idx:
                raise ValueError(f"target_node='{target_node}' not found. Available: {self.node_names}")
            self.target_idx = self.name_to_idx[target_node]

        self.desired_state = int(desired_state)
        target_card = self.cardinalities[self.target_idx]
        if not (0 <= self.desired_state < target_card):
            raise ValueError(f"desired_state={self.desired_state} out of range for target card={target_card}")

        # Allowed intervention nodes -> indices
        if allowed_intervention_nodes is None:
            self.allowed_nodes = list(range(self.n_nodes))
        else:
            for n in allowed_intervention_nodes:
                if n not in self.name_to_idx:
                    raise ValueError(f"allowed_intervention_nodes contains unknown node '{n}'.")
            self.allowed_nodes = [self.name_to_idx[n] for n in allowed_intervention_nodes]

        # Topological order (as names)
        self._topo = _topo_order_from_edges(self.node_names, list(self.model.edges()))

        # Generate queries
        self.queries = self._generate_pairwise_data(self.n_queries)

    # ---------------- Public API ----------------
    def get_causal_graph(self):
        return self.adj.clone(), list(self.node_names)

    def __len__(self):
        return self.n_queries

    def __getitem__(self, idx):
        return self.queries[idx]

    # ---------------- PCBO sampling ----------------
    def _sample_intervention(self):
        # choose node index from allowed_nodes
        j = int(torch.randint(0, len(self.allowed_nodes), (1,), generator=self._torch_rng).item())
        node_idx = int(self.allowed_nodes[j])

        card = int(self.cardinalities[node_idx])
        val = int(self._np_rng.integers(0, card))
        return node_idx, val

    def _true_utility(self, outcome_vec: torch.Tensor, intervention_node: int | None):
        # outcome_vec stores integer-coded states as float32
        u = -torch.abs(outcome_vec[self.target_idx] - float(self.desired_state))
        if intervention_node is not None:
            u = u - self.intervention_cost
        return u

    def _sample_assignment_do(self, do_node_idx: int, do_value: int) -> np.ndarray:
        """
        Ancestral sampling under hard do(X_do=do_value).
        """
        do_node = self.node_names[do_node_idx]
        assn = {n: None for n in self.node_names}
        assn[do_node] = int(do_value)

        for node in self._topo:
            if assn[node] is not None:
                continue

            cpd = self.cpds[node]
            # IMPORTANT: pgmpy convention: cpd.variables = [child, parent1, parent2, ...]
            parents = list(cpd.variables[1:])

            if parents:
                parent_vals = []
                for p in parents:
                    pv = assn[p]
                    if pv is None:
                        # if topo order is correct, this shouldn't happen;
                        # fallback: sample parent first from its marginal
                        parent_cpd = self.cpds[p]
                        pv = _sample_from_cpd_values(parent_cpd.values, None, self._np_rng)
                        assn[p] = int(pv)

                    # sanity: check within parent cardinality
                    p_idx = self.name_to_idx[p]
                    p_card = self.cardinalities[p_idx]
                    if not (0 <= int(pv) < int(p_card)):
                        raise RuntimeError(f"Invalid parent state: {p}={pv} but card={p_card}")

                    parent_vals.append(int(pv))

                x = _sample_from_cpd_values(cpd.values, parent_vals, self._np_rng)
            else:
                x = _sample_from_cpd_values(cpd.values, None, self._np_rng)

            assn[node] = int(x)

        x_vec = np.array([assn[n] for n in self.node_names], dtype=np.int64)
        return x_vec

    def _generate_pairwise_data(self, n_queries: int):
        qs = []
        for _ in range(n_queries):
            intervs, outs, utils = [], [], []

            for _ in range(2):
                node, val = self._sample_intervention()
                x = self._sample_assignment_do(node, val)
                out = torch.tensor(x, dtype=torch.float32)
                util = self._true_utility(out, node)

                intervs.append((node, val))
                outs.append(out)
                utils.append(util)

            outs = torch.stack(outs, dim=0)
            utils = torch.stack(utils, dim=0).squeeze(-1)
            winner_idx = int(torch.argmax(utils).item())

            qs.append({
                "interventions": intervs,
                "outcomes": outs,
                "utilities": utils,
                "winner_idx": winner_idx,
            })

        return qs
    
    def _compute_intervention_outcome(self, node_idx: int, value):
        """
        Compatibility method with your existing PCBO code.

        For BIF datasets, interventions are discrete:
            do(X_node_idx = state_index)

        If value is a float (e.g., -1.5, 0.0, 1.5), we map it to a valid state index.
        """
        card = int(self.cardinalities[int(node_idx)])

        # Map continuous values to discrete states if needed
        if isinstance(value, (float, np.floating)) or (isinstance(value, torch.Tensor) and value.ndim == 0):
            v = float(value)
            # simple robust mapping: quantize into {0, ..., card-1}
            if card == 1:
                state = 0
            else:
                # map [-inf, +inf] -> [0, card-1] via tanh then scaling
                z = np.tanh(v)  # [-1, 1]
                state = int(np.round((z + 1.0) * 0.5 * (card - 1)))
                state = max(0, min(card - 1, state))
        else:
            # assume it's already a discrete state index
            state = int(value)
            state = max(0, min(card - 1, state))

        x = self._sample_assignment_do(int(node_idx), int(state))  # np int64 (d,)
        return torch.tensor(x, dtype=torch.float32)  # torch float32 (d,)