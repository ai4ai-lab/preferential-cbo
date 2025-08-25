import torch
import numpy as np
from typing import Tuple
import math


class ScalableParentPosterior:
    """
    MCMC-based parent posterior for scalable causal discovery.
    Instead of enumerating all 2^d subsets, we use MCMC to sample from the posterior.
    """
    
    def __init__(
        self,
        d: int,
        a0: float = 1.0,
        b0: float = 1.0,
        tau2: float = 1.0,
        prior_sparsity: float = 0.2,
        max_parents: int = 5,  # Maximum number of parents to consider
        device: torch.device = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.d = d
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.tau2 = float(tau2)
        self.pi = float(prior_sparsity)
        self.max_parents = min(max_parents, d)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        
        # Sufficient statistics
        self.reset()
        
        # MCMC state
        self.current_mask = torch.zeros(d, dtype=torch.bool, device=device)
        self.current_log_prob = -float("inf")
        self.mcmc_samples = []
        self.mcmc_log_probs = []
        
    def reset(self):
        """Reset sufficient statistics."""
        self.n = 0
        self.XtX = torch.zeros((self.d, self.d), dtype=self.dtype, device=self.device)
        self.Xty = torch.zeros((self.d, 1), dtype=self.dtype, device=self.device)
        self.yty = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        
    def add_datapoint(self, x, y):
        """Add observed (x, y) to sufficient statistics."""
        x = x.to(self.device, self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = y.to(self.device, self.dtype).reshape(-1, 1)
        
        self.XtX += x.T @ x
        self.Xty += x.T @ y
        self.yty += (y.T @ y).squeeze()
        self.n += 1
        
    def _log_prior_mask(self, mask):
        """Log prior for a parent set mask."""
        k = mask.sum().item()
        if k > self.max_parents:
            return -float("inf")
        return k * math.log(self.pi) + (self.d - k) * math.log(1.0 - self.pi)
    
    def _log_marginal_likelihood(self, mask):
        """Compute log p(y | X_S) for given mask S."""
        idx = torch.where(mask)[0]
        p = len(idx)
        
        log_prior = self._log_prior_mask(mask)
        if math.isinf(log_prior):
            return log_prior
            
        if self.n == 0:
            return log_prior
            
        if p == 0:
            # No parents case
            a_n = self.a0 + 0.5 * self.n
            b_n = self.b0 + 0.5 * self.yty.item()
            log_ml = (
                self.a0 * math.log(self.b0)
                - a_n * math.log(b_n)
                + torch.lgamma(torch.tensor(a_n)).item()
                - torch.lgamma(torch.tensor(self.a0)).item()
            )
        else:
            XtX_S = self.XtX[idx][:, idx]
            Xty_S = self.Xty[idx, :]
            
            V0_inv = (1.0 / self.tau2) * torch.eye(p, dtype=self.dtype, device=self.device)
            Vn_inv = V0_inv + XtX_S
            
            try:
                L = torch.linalg.cholesky(Vn_inv)
                logdet_Vn = -2.0 * torch.sum(torch.log(torch.diag(L)))
                u = torch.cholesky_solve(Xty_S, L)
                quad_term = (Xty_S.T @ u).item()
            except:
                return -float("inf")
                
            logdet_V0 = p * math.log(self.tau2)
            log_det_ratio = logdet_V0 - logdet_Vn
            
            a_n = self.a0 + 0.5 * self.n
            b_n = self.b0 + 0.5 * (self.yty.item() - quad_term)
            
            log_ml = (
                0.5 * log_det_ratio
                + self.a0 * math.log(self.b0)
                - a_n * math.log(max(b_n, 1e-10))
                + torch.lgamma(torch.tensor(a_n)).item()
                - torch.lgamma(torch.tensor(self.a0)).item()
            )
            
        return log_prior + log_ml
    
    def _propose_move(self, current_mask):
        """Propose a new mask by flipping one bit."""
        mask = current_mask.clone()
        idx = torch.randint(0, self.d, (1,)).item()
        mask[idx] = ~mask[idx]
        
        # Ensure we don't exceed max parents
        if mask.sum() > self.max_parents:
            mask[idx] = False
            
        return mask
    
    def run_mcmc(self, n_samples = 2000, burn_in = 500, thin = 5):
        """Tempered MCMC with warm-start and guardrails."""
        self.mcmc_samples, self.mcmc_log_probs = [], []

        # ---- init mask
        if getattr(self, "best_mask", None) is not None:
            current_mask = self.best_mask.clone()
        else:
            if hasattr(self, "XtX") and hasattr(self, "Xty") and hasattr(self, "yty") and self.d > 0:
                XtX_diag = torch.sqrt(torch.clamp(torch.diag(self.XtX), min=1e-8))
                denom = XtX_diag * torch.sqrt(torch.clamp(self.yty, min=1e-8))
                corr = (self.Xty.view(-1) / denom).clamp(-1, 1)
                mask = torch.zeros(self.d, dtype=torch.bool, device=self.device)
                k = min(int(self.max_parents), self.d)
                topk = torch.topk(torch.abs(corr), k).indices
                mask[topk] = True
                current_mask = mask
            else:
                current_mask = torch.bernoulli(torch.full((self.d,), self.pi, device=self.device)).bool()

        current_lp = self._log_marginal_likelihood(current_mask)

        # ---- parallel tempering
        temperatures = [1.0, 1.5, 2.25, 3.4]  # geometric
        chains = [current_mask.clone() for _ in temperatures]
        lps = [current_lp.clone() for _ in temperatures]
        acc_moves = [0] * len(temperatures)
        tot_moves = [0] * len(temperatures)

        total_iters = n_samples + burn_in
        for i in range(total_iters):
            # within-chain updates
            for t_idx, T in enumerate(temperatures):
                prop = self._propose_move(chains[t_idx])  # must enforce constraints
                prop_lp = self._log_marginal_likelihood(prop)
                log_ratio = float((prop_lp - lps[t_idx]) / T)
                accept = (log_ratio >= 0.0) or (torch.rand(1, device=self.device).item() < math.exp(log_ratio))
                tot_moves[t_idx] += 1
                if accept:
                    chains[t_idx], lps[t_idx] = prop, prop_lp
                    acc_moves[t_idx] += 1

            # occasional swaps
            if i % 10 == 0:
                for t_idx in range(len(temperatures) - 1):
                    delta = float((lps[t_idx + 1] - lps[t_idx]) * (1/temperatures[t_idx] - 1/temperatures[t_idx + 1]))
                    swap_ok = (delta >= 0.0) or (torch.rand(1, device=self.device).item() < math.exp(delta))
                    if swap_ok:
                        chains[t_idx], chains[t_idx + 1] = chains[t_idx + 1], chains[t_idx]
                        lps[t_idx], lps[t_idx + 1] = lps[t_idx + 1], lps[t_idx]

            # save from the cold chain
            if i >= burn_in and ((i - burn_in) % thin == 0):
                self.mcmc_samples.append(chains[0].clone())
                self.mcmc_log_probs.append(float(lps[0]))

        # warm-start for next call
        if self.mcmc_log_probs:
            best_idx = int(np.argmax(self.mcmc_log_probs))
            self.best_mask = self.mcmc_samples[best_idx].clone()
    
    def edge_posterior(self, n_mcmc = 500):
        """Compute edge posteriors via MCMC."""
        if self.n < 2:
            return torch.full((self.d,), 0.5, dtype=self.dtype, device=self.device)
            
        self.run_mcmc(n_samples=n_mcmc, burn_in=100)
        
        edge_probs = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        for mask in self.mcmc_samples:
            edge_probs += mask.float()
        edge_probs /= len(self.mcmc_samples)
        
        return edge_probs

    def most_probable_set(self):
        """Return MAP estimate from MCMC samples."""
        if not self.mcmc_samples:
            self.run_mcmc(n_samples=500, burn_in=100)
            
        if not self.mcmc_samples:
            return torch.zeros(self.d, dtype=torch.bool, device=self.device), 0.0
            
        best_idx = np.argmax(self.mcmc_log_probs)
        best_mask = self.mcmc_samples[best_idx]
        best_prob = math.exp(self.mcmc_log_probs[best_idx] - max(self.mcmc_log_probs))
        
        return best_mask, best_prob
    
    def update_posterior(self):
        """Trigger MCMC resampling with current data."""
        # Reset MCMC state to encourage exploration
        self.current_log_prob = -float("inf")
        
        
class ScalableLocalParentPosterior(ScalableParentPosterior):
    """Local version that handles slicing for specific target nodes."""
    
    def __init__(self, parent_idx, target_idx, **kwargs):
        # Remove 'd' from kwargs if present since we calculate it
        kwargs.pop('d', None)
        super().__init__(d=len(parent_idx), **kwargs)
        self.parent_idx = list(parent_idx)
        self.target_idx = int(target_idx)
        
    def slice_xy(self, outcome):
        """Extract (x_parents, y_target) from full outcome."""
        out = outcome.to(self.device, self.dtype).view(-1)
        x = out[self.parent_idx].unsqueeze(0)
        y = out[self.target_idx].view(1, 1)
        return x, y
    
    def add_datapoint_full(self, outcome, do_idx):
        """Add data unless we intervened on target."""
        if do_idx == self.target_idx:
            return
        x, y = self.slice_xy(outcome)
        super().add_datapoint(x, y)
        
    def num_datapoints(self):
        return int(self.n)
    
    @torch.no_grad()
    def peek_update_edge_posterior(self, x_new, y_new):
        """
        Return edge posterior vector AFTER a virtual update with (x_new, y_new).
        """
        x = x_new.to(self.device, self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = y_new.to(self.device, self.dtype).reshape(-1, 1)
        
        # Store current stats
        old_XtX = self.XtX.clone()
        old_Xty = self.Xty.clone()
        old_yty = self.yty.clone()
        old_n = self.n
        
        # Temporarily update
        self.XtX = old_XtX + x.T @ x
        self.Xty = old_Xty + x.T @ y
        self.yty = old_yty + (y.T @ y).squeeze()
        self.n = old_n + 1
        
        # Get posterior with new data
        edge_probs = self.edge_posterior(n_mcmc=200)
        
        # Restore original stats
        self.XtX = old_XtX
        self.Xty = old_Xty
        self.yty = old_yty
        self.n = old_n
        
        return edge_probs
    
    @torch.no_grad()
    def peek_update_edge_entropy(self, x_new, y_new):
        """
        Return sum_j H(edge_j | data & {(x_new,y_new)}).
        """
        pj = self.peek_update_edge_posterior(x_new, y_new)
        eps = 1e-12
        pj = torch.clamp(pj, eps, 1.0 - eps)
        H = -(pj * torch.log(pj) + (1 - pj) * torch.log(1 - pj))
        return H.sum()
    
    @torch.no_grad()
    def peek_update_edge_posterior_full(self, outcome, intervened_idx=None):
        """Full outcome version of peek_update_edge_posterior."""
        if intervened_idx == self.target_idx:
            return self.edge_posterior()
        x, y = self.slice_xy(outcome)
        return self.peek_update_edge_posterior(x, y)
    
    @torch.no_grad()
    def peek_update_edge_entropy_full(self, outcome, intervened_idx=None):
        """Full outcome version of peek_update_edge_entropy."""
        if intervened_idx == self.target_idx:
            pj = self.edge_posterior()
            eps = 1e-12
            pj = torch.clamp(pj, eps, 1.0 - eps)
            H = -(pj * torch.log(pj) + (1 - pj) * torch.log(1 - pj))
            return H.sum()
        x, y = self.slice_xy(outcome)
        return self.peek_update_edge_entropy(x, y)