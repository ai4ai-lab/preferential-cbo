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

        if node_idx == 0:      # X1
            x1 = torch.as_tensor(value)  # Value can be float or tensor; this makes it tensor
        elif node_idx == 1:    # X2
            x2 = torch.as_tensor(value)
        elif node_idx == 2:    # Y (hard intervention)
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