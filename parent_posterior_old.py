import torch
import torch.nn.functional as F
import itertools
import numpy as np

class ParentPosterior:
    def __init__(self, d, sigma_eps=0.1, sigma_theta=1.0, prior_sparsity=0.5):
        """
        Bayesian posterior over all 2^d subsets of d candidate parents for a target Y.

        Parameters:
        - d (int): Number of candidate covariates (X1, ..., X_d)
        - sigma_eps (float): Noise std dev in observation model Y = Xθ + ε
        - sigma_theta (float): Prior std dev for θ ~ N(0, σ² I)
        - prior_sparsity (float): Bernoulli prior over inclusion of each parent
        """
        self.d = d
        # Linear-regression hyper‐parameters
        self.sigma_eps = sigma_eps
        self.sigma_theta = sigma_theta
        
        # Enumerate all 2^d parent sets (binary masks)
        self.S = list(itertools.product([0,1], repeat=d))
        self.M = len(self.S)

        # Compute prior log-probabilities for each parent set
        # log prior P(G) -> log P(G) = log P(sparsity)^|S| * log(1-P(sparsity))^(d-|S|)
        logp0 = np.array([
            np.sum(s) * np.log(prior_sparsity) + (d - np.sum(s)) * np.log(1 - prior_sparsity)
            for s in self.S
        ])
        self.log_post = torch.tensor(logp0, dtype=torch.float64)

        # Store sufficient statistics (X^T X, X^T y), needed for marginal likelihood (Bayesian linear regression)
        self.XTX = torch.zeros((d,d), dtype=torch.float64)
        self.XTy = torch.zeros(d, dtype=torch.float64)
        self.n = 0
        self.ssq_y = 0.0  # sum of squares of y

    def add_datapoint(self, x, y):
        """
        Incorporate one (x, y) observation into the sufficient statistics.

        Parameters:
        - x (1 x d tensor): Covariates
        - y (float): Target variable
        """
        x = x.double()
        self.XTX += x.T @ x  # outer product, (dxd)
        self.XTy += x.flatten() * y  # inner product, (dx1)
        self.ssq_y += float(y ** 2)  # accumulate sum of squares
        self.n += 1

    def _log_marginal_likelihood(self):
        """
        Compute marginal likelihood log p(y | X_S) for each parent set S under a conjugate
        Normal-Inverse-Gamma model with known noise variance (sigma_eps^2).
        Works for p = 0 (empty set) as well.

        Returns:
        - logml (tensor): Log marginal likelihood for each of the M parent sets.
        """
        logml = torch.empty(self.M, dtype=torch.float64)

        for idx, s in enumerate(self.S):
            mask = torch.tensor(s, dtype=torch.bool)
            p = mask.sum().item()  # number of parents in this set

            if p == 0:  # empty parent set
                logml[idx] = -self.n/2 * np.log(2*np.pi*self.sigma_eps**2) - self.ssq_y / (2*self.sigma_eps**2)
            else:
                XTXs = self.XTX[mask][:,mask]
                XTy = self.XTy[mask]
                # Bayesian linear regression closed form
                # log P(Y | X_s) = -n/2 * log(2*pi*sigma_eps^2) + 0.5 * log det(Sigma_post) / det(Sigma0_inv) + 0.5 * (XTy @ mu_post) / sigma_eps^2
                Sigma0_inv = torch.eye(p, dtype=torch.float64) / self.sigma_theta**2
                Sigma_post = torch.inverse(Sigma0_inv + XTXs / self.sigma_eps**2)
                mu_post = Sigma_post @ (XTy / self.sigma_eps**2)

                quad = XTy @ mu_post
                log_det = torch.logdet(Sigma_post) - torch.logdet(Sigma0_inv)

                logml[idx] = (-self.n/2 * np.log(2*np.pi*self.sigma_eps**2)
                    + 0.5 * log_det
                    + 0.5 * quad / (self.sigma_eps**2)
                )

        return logml

    def update_posterior(self):
        """
        Update log posterior distribution over parent sets.
        """
        logml = self._log_marginal_likelihood()
        self.log_post += logml
        # Normalize in log-space (avoid numerical issues)
        self.log_post -= torch.logsumexp(self.log_post, dim=0)

    def edge_posterior(self):
        """
        Return P(X_j -> Y) for each covariate j.

        Returns:
        - probs (d-tensor): Marginal posterior probability of each edge.
        """
        post = torch.exp(self.log_post)
        probs = torch.zeros(self.d, dtype=torch.float64)
        for idx, s in enumerate(self.S):
            probs += post[idx] * torch.tensor(s, dtype=torch.float64)
        return probs

    def most_probable_set(self):
        """
        Return the most probable parent set and its posterior probability.

        Returns:
        - tuple: (most probable parent set as tensor, posterior probability)
        """
        idx = torch.argmax(self.log_post)
        return self.S[int(idx)], torch.exp(self.log_post[idx]).item()
    

    def sample_graph(self, n_samples=1):
        """
        Sample graphs from the posterior distribution
        
        Returns:
            List of adjacency matrices
        """
        probs = torch.exp(self.log_post)
        indices = torch.multinomial(probs, n_samples, replacement=True)
        
        graphs = []
        for idx in indices:
            parent_set = self.S[idx]
            # Convert to adjacency matrix (only edges to Y)
            adj = torch.zeros((self.d + 1, self.d + 1))
            for i, is_parent in enumerate(parent_set):
                if is_parent:
                    adj[i, self.d] = 1  # Edge from Xi to Y
            graphs.append(adj)
        
        return graphs

    def expected_value_of_information(self, intervention_node, intervention_value):
        """
        Compute expected reduction in entropy from intervening
        
        This is key for active causal discovery
        """
        current_entropy = self.entropy()
        
        # Simulate potential outcomes under current graph beliefs
        expected_entropy = 0.0
        
        for idx, parent_set in enumerate(self.S):
            prob = torch.exp(self.log_post[idx])
            
            # Predict outcome under this graph
            # This is simplified - you'd need the actual SCM parameters
            predicted_outcome = self._predict_outcome(intervention_node, 
                                                    intervention_value, 
                                                    parent_set)
            
            # Compute posterior entropy if we observed this outcome
            future_entropy = self._posterior_entropy_given_outcome(
                intervention_node, intervention_value, predicted_outcome
            )
            
            expected_entropy += prob * future_entropy
        
        return current_entropy - expected_entropy

    def entropy(self):
        """Compute entropy of current posterior"""
        probs = torch.exp(self.log_post)
        return -(probs * self.log_post).sum()

    def integrate_with_utility(self, utility_function):
        """
        Update posterior using preference data through utility function
        
        This is where causal discovery meets preference learning!
        """
        # Connecting the utility function to the graph structure
        # Implementation depends on how preferences constrain graphs
        pass