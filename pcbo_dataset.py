import torch
import numpy as np
from torch.utils.data import Dataset

class PCBO_SyntheticDataset(Dataset):
    def __init__(self, n_queries=100, k=3, d=2, noise_std=0.1, domain=(-2, 2), seed=42):
        """
        Generate k-wise preference data based on interventions on a simple SCM: X1 -> Y <- X2
        """
        super().__init__()
        self.n_queries = n_queries
        self.k = k
        self.d = d
        self.noise_std = noise_std
        self.domain = domain
        self.seed = seed
        self.queries = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Node names for the graph visualisation
        self.node_names = [f"X{i}" for i in range(self.d)] + ["Y"]
        # For toy X1 -> Y <- X2 (2 inputs + Y = 3 nodes)
        self.node_names = ["X1", "X2", "Y"]
        # Adjacency matrix for the graph X1 -> Y <- X2
        self.adj = torch.tensor([[0,0,1],  # X1 -> Y
                                [0,0,1],  # X2 -> Y
                                [0,0,0]], dtype=torch.int)

        self._generate_preferences()

    def scm(self, x):
        """SCM equation (non-linear but smooth): Y = tanh(x1) + 0.5 * sin(x2) + noise"""
        x1 = x[:, 0]
        x2 = x[:, 1]
        y = torch.tanh(x1) + 0.5 * torch.sin(x2)
        y += torch.randn_like(y) * self.noise_std
        return y

    def _generate_preferences(self):
        torch.manual_seed(self.seed)
        for _ in range(self.n_queries):
            # Sample k interventions uniformly in the domain (each is a 2D point)
            x_choices = (torch.rand(self.k, self.d) * (self.domain[1] - self.domain[0]) + self.domain[0])
            # SCM outputs for these choices
            y_values = self.scm(x_choices)
            winner_index = torch.argmin(y_values)  # best = lowest Y
            self.queries.append((x_choices, winner_index))

    def _generate_one_query(self):
        """Return a single k-wise query. Same logic as in _generate_preferences but for one query only, returns y_values as well.
        This is useful for generating new queries on-the-fly during training.
        """
        x_choices = (torch.rand(self.k, self.d) * (self.domain[1] - self.domain[0]) + self.domain[0])
        y_values = self.scm(x_choices)
        winner_idx = int(torch.argmin(y_values).item())
        return x_choices, winner_idx, y_values

    def __len__(self):
        return self.n_queries

    def __getitem__(self, idx):
        return self.queries[idx]