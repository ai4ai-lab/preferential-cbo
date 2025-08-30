import torch
import numpy as np
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
        tau2_env: float = 10.0,
    ):
        self.d = d
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.tau2 = float(tau2)
        self.pi = float(prior_sparsity)
        self.max_parents = min(max_parents, d)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.tau2_env = float(tau2_env)

        self.d_env = 0
        self.reset()
        
        # MCMC state
        self.current_mask = torch.zeros(d, dtype=torch.bool, device=self.device)
        self.current_log_prob = -float("inf")
        self.mcmc_samples = []
        self.mcmc_log_probs = []

        self._edge_prob_cache = None
        self._cache_valid = False

    def reset(self):
        """Reset sufficient statistics."""
        self.n = 0
        D_tot = self.d + self.d_env
        self.XtX = torch.zeros((D_tot, D_tot), dtype=self.dtype, device=self.device)
        self.Xty = torch.zeros((D_tot, 1), dtype=self.dtype, device=self.device)
        self.yty = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    def _expand_env(self, k):
        """Dynamically add k environment columns to the sufficient stats."""
        if k <= 0:
            return
        old_env = self.d_env
        new_env = old_env + k
        old_D = self.d + old_env
        new_D = self.d + new_env

        XtX_new = torch.zeros((new_D, new_D), dtype=self.dtype, device=self.device)
        Xty_new = torch.zeros((new_D, 1), dtype=self.dtype, device=self.device)
        if old_D > 0:
            XtX_new[:old_D, :old_D] = self.XtX
            Xty_new[:old_D, :] = self.Xty
        self.XtX = XtX_new
        self.Xty = Xty_new
        # yty, n unchanged
        self.d_env = new_env
        
    def add_datapoint(self, x, y, x_env=None):
        """Add observed (x_parents, y) with optional environment one-hot x_env."""
        x = x.to(self.device, self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = y.to(self.device, self.dtype).reshape(-1, 1)

        if x_env is not None:
            x_env = x_env.to(self.device, self.dtype)
            if x_env.dim() == 1:
                x_env = x_env.unsqueeze(0)
            need_env = x_env.size(1)
            if need_env > self.d_env:
                self._expand_env(need_env - self.d_env)
            z = torch.cat([x, x_env], dim=1)
        else:
            if self.d_env > 0:
                pad = torch.zeros((x.size(0), self.d_env), dtype=self.dtype, device=self.device)
                z = torch.cat([x, pad], dim=1)
            else:
                z = x

        self.XtX += z.T @ z
        self.Xty += z.T @ y
        self.yty += (y.T @ y).squeeze()
        self.n += 1

        self._cache_valid = False  # Invalidate cache when data changes
        
    def _log_prior_mask(self, mask):
        """Log prior for a parent set mask."""
        k = mask.sum().item()
        if k > self.max_parents:
            return -float("inf")
        return k * math.log(self.pi) + (self.d - k) * math.log(1.0 - self.pi)
    
    def _log_marginal_likelihood(self, mask):
        """Compute log p(y | X_S, env) for given parent mask S, env always included."""
        # prior over parents (respect max_parents)
        log_prior = self._log_prior_mask(mask)
        if math.isinf(log_prior):
            return log_prior
        if self.n == 0:
            return log_prior

        # infer how many env columns exist in stats
        D_tot = self.XtX.size(0)
        d_env = max(0, D_tot - self.d)

        idx_par = torch.where(mask)[0].tolist()
        idx_env = list(range(self.d, self.d + d_env))
        idx_all = idx_par + idx_env
        p = len(idx_all)

        if p == 0:
            a_n = self.a0 + 0.5 * self.n
            b_n = self.b0 + 0.5 * self.yty.item()
            log_ml = (
                self.a0 * math.log(self.b0)
                - a_n * math.log(b_n)
                + torch.lgamma(torch.tensor(a_n, dtype=self.dtype)).item()
                - torch.lgamma(torch.tensor(self.a0, dtype=self.dtype)).item()
            )
            return log_prior + log_ml

        idx = torch.tensor(idx_all, dtype=torch.long, device=self.device)
        XtX_S = self.XtX[idx][:, idx]
        Xty_S = self.Xty[idx, :]

        n_par = len(idx_par)
        n_env = len(idx_env)

        prior_prec = torch.full((p,), 1.0 / self.tau2, dtype=self.dtype, device=self.device)
        if n_env > 0:
            prior_prec[-n_env:] = 1.0 / self.tau2_env

        Vn_inv = XtX_S + torch.diag(prior_prec)

        try:
            L = torch.linalg.cholesky(Vn_inv)
            logdet_Vn = -2.0 * torch.sum(torch.log(torch.diag(L)))
            u = torch.cholesky_solve(Xty_S, L)
            quad_term = (Xty_S.T @ u).item()
        except Exception:
            return -float("inf")

        logdet_V0 = n_par * math.log(self.tau2) + n_env * math.log(self.tau2_env)
        log_det_ratio = logdet_V0 - logdet_Vn

        a_n = self.a0 + 0.5 * self.n
        b_n = self.b0 + 0.5 * max(self.yty.item() - quad_term, 1e-12)

        log_ml = (
            0.5 * log_det_ratio
            + self.a0 * math.log(self.b0)
            - a_n * math.log(b_n)
            + torch.lgamma(torch.tensor(a_n, dtype=self.dtype)).item()
            - torch.lgamma(torch.tensor(self.a0, dtype=self.dtype)).item()
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
                # parent-only diagonals / cross terms
                XtX_par = self.XtX[:self.d, :self.d]
                Xty_par = self.Xty[:self.d, :]
                XtX_diag = torch.sqrt(torch.clamp(torch.diag(XtX_par), min=1e-8))
                denom = XtX_diag * torch.sqrt(torch.clamp(self.yty, min=1e-8))
                corr = (Xty_par.view(-1) / denom).clamp(-1, 1)
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

    def edge_posterior(self, n_mcmc=200, force_recompute=False):
        """Compute edge posteriors via MCMC."""
        if self._cache_valid and not force_recompute:
            return self._edge_prob_cache
            
        # Only run MCMC if cache invalid
        self.run_mcmc(n_samples=n_mcmc, burn_in=100)
        
        edge_probs = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        for mask in self.mcmc_samples:
            edge_probs += mask.float()
        edge_probs /= len(self.mcmc_samples)
        
        self._edge_prob_cache = edge_probs
        self._cache_valid = True
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

    @torch.no_grad()
    def peek_update_edge_posterior_aug(self, x_par, y, x_env=None, n_mcmc=50):
        """
        Edge posterior after a virtual update with (x_par, y, x_env), without mutating state.
        """
        x_par = x_par.to(self.device, self.dtype)
        if x_par.dim() == 1:
            x_par = x_par.unsqueeze(0)
        y = y.to(self.device, self.dtype).reshape(-1, 1)

        # determine target env size
        need_env = self.d_env
        if x_env is not None:
            x_env = x_env.to(self.device, self.dtype)
            if x_env.dim() == 1:
                x_env = x_env.unsqueeze(0)
            need_env = max(need_env, x_env.size(1))

        old_D = self.XtX.size(0)
        new_D = self.d + need_env
        XtX_use = self.XtX
        Xty_use = self.Xty
        if new_D > old_D:
            XtX_pad = torch.zeros((new_D, new_D), dtype=self.dtype, device=self.device)
            Xty_pad = torch.zeros((new_D, 1), dtype=self.dtype, device=self.device)
            if old_D > 0:
                XtX_pad[:old_D, :old_D] = self.XtX
                Xty_pad[:old_D, :] = self.Xty
            XtX_use, Xty_use = XtX_pad, Xty_pad

        if x_env is None:
            x_env = torch.zeros((1, new_D - self.d), dtype=self.dtype, device=self.device)

        z = torch.cat([x_par, x_env], dim=1)
        XtX_new = XtX_use + z.T @ z
        Xty_new = Xty_use + z.T @ y
        yty_new = self.yty + (y.T @ y).squeeze()
        n_new = self.n + 1

        # Temporarily run MCMC using the temporary stats
        # Save originals
        XtX_old, Xty_old, yty_old, n_old = self.XtX, self.Xty, self.yty, self.n
        d_env_old = self.d_env
        # Swap in
        self.XtX, self.Xty, self.yty, self.n = XtX_new, Xty_new, yty_new, n_new
        self.d_env = need_env
        try:
            pj = self.edge_posterior(n_mcmc=n_mcmc)
        finally:
            # Restore
            self.XtX, self.Xty, self.yty, self.n = XtX_old, Xty_old, yty_old, n_old
            self.d_env = d_env_old
        return pj

    @torch.no_grad()
    def peek_update_edge_posterior(self, x_new, y_new, x_env=None):
        return self.peek_update_edge_posterior_aug(x_new, y_new, x_env=x_env, n_mcmc=200)

    @torch.no_grad()
    def peek_update_edge_entropy(self, x_new, y_new, x_env=None):
        pj = self.peek_update_edge_posterior_aug(x_new, y_new, x_env=x_env, n_mcmc=200)
        eps = 1e-12
        pj = torch.clamp(pj, eps, 1.0 - eps)
        H = -(pj * torch.log(pj) + (1 - pj) * torch.log(1 - pj))
        return H.sum()
        
        
class ScalableLocalParentPosterior(ScalableParentPosterior):
    """Local version that handles slicing for specific target nodes."""
    
    def __init__(self, parent_idx, target_idx, **kwargs):
        # Remove 'd' from kwargs if present since we calculate it
        kwargs.pop('d', None)
        super().__init__(d=len(parent_idx), **kwargs)
        self.parent_idx = list(parent_idx)
        self.target_idx = int(target_idx)
        self.env_map = {}  # global intervened node -> local env column index
        
    def slice_xy(self, outcome):
        """Extract (x_parents, y_target) from full outcome."""
        out = outcome.to(self.device, self.dtype).view(-1)
        x = out[self.parent_idx].unsqueeze(0)
        y = out[self.target_idx].view(1, 1)
        return x, y
    
    def add_datapoint_full(self, outcome, do_idx):
        """Add data unless we intervened on target (include env one-hot)."""
        if do_idx == self.target_idx:
            return
        x, y = self.slice_xy(outcome)
        x_env = self._x_env_onehot(int(do_idx))
        super().add_datapoint(x, y, x_env=x_env)
        
    def num_datapoints(self):
        return int(self.n)
    
    def _x_env_onehot(self, do_idx: int):
        """
        Return (1, d_env) one-hot for the intervened node (excluding the target).
        Dynamically allocates a new env column for a new 'do' node.
        """
        if do_idx == self.target_idx:
            return torch.zeros((1, self.d_env), dtype=self.dtype, device=self.device)

        if do_idx not in self.env_map:
            super()._expand_env(1)
            self.env_map[do_idx] = self.d_env - 1

        v = torch.zeros((1, self.d_env), dtype=self.dtype, device=self.device)
        v[0, self.env_map[do_idx]] = 1.0
        return v
    
    @torch.no_grad()
    def peek_update_edge_posterior(self, x_new, y_new, x_env=None):
        """
        Return edge posterior AFTER a virtual update with (x_new, y_new, x_env),
        using the base class's robust augmented logic (handles env padding).
        """
        return self.peek_update_edge_posterior_aug(x_new, y_new, x_env=x_env, n_mcmc=200)

    @torch.no_grad()
    def peek_update_edge_entropy(self, x_new, y_new, x_env=None):
        """
        Return sum_j H(edge_j | data ∪ {(x_new, y_new, x_env)}).
        """
        pj = self.peek_update_edge_posterior_aug(x_new, y_new, x_env=x_env, n_mcmc=200)
        eps = 1e-12
        pj = torch.clamp(pj, eps, 1.0 - eps)
        H = -(pj * torch.log(pj) + (1 - pj) * torch.log(1 - pj))
        return H.sum()
    
    @torch.no_grad()
    def peek_update_edge_posterior_full(self, outcome, intervened_idx=None):
        if intervened_idx == self.target_idx:
            return self.edge_posterior()
        x, y = self.slice_xy(outcome)
        x_env = self._x_env_onehot(int(intervened_idx)) if intervened_idx is not None else None
        return self.peek_update_edge_posterior_aug(x, y, x_env=x_env, n_mcmc=200)

    @torch.no_grad()
    def peek_update_edge_entropy_full(self, outcome, intervened_idx=None):
        if intervened_idx == self.target_idx:
            pj = self.edge_posterior()
            eps = 1e-12
            pj = torch.clamp(pj, eps, 1.0 - eps)
            H = -(pj * torch.log(pj) + (1 - pj) * torch.log(1 - pj))
            return H.sum()
        x, y = self.slice_xy(outcome)
        x_env = self._x_env_onehot(int(intervened_idx)) if intervened_idx is not None else None
        return self.peek_update_edge_entropy(x, y, x_env=x_env)