import torch
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