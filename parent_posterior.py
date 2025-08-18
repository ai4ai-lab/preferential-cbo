import math
from typing import Tuple, List
import numpy as np
import torch


class ParentPosterior:
    """
    Bayesian posterior over parent subsets S in {1,...,d} for local linear-Gaussian model of Y:
        y = X_S beta_S + eps, with eps ~ N(0, sigma^2 I)
        beta_S | sigma^2 ~ N(0, sigma^2 tau^2 I)
        sigma^2 ~ InvGamma(a0, b0)
    Prior over subsets factorizes: p(S) proportional to prod_j pi^{z_j} (1-pi)^{1-z_j}

    Sufficient statistics are accumulated ONLY from executed interventions:
        XtX = sum x x^T,  Xty = sum x y,  yty = sum y^2,  n = #points
    """
    def __init__(
        self,
        d,
        a0 = 1.0,  # Inverse Gamma prior shape
        b0 = 1.0,  # Inverse Gamma prior scale
        tau2 = 1.0,  # prior scale for beta (ridge)
        prior_sparsity = 0.2,  # Bernoulli(pi) per edge
        prior_mode = "bernoulli",  # "bernoulli" or "uniform"
        a_pi = 1.0,  # prior shape for pi
        b_pi = 1.0,  # prior scale for pi
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.d = d
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.tau2 = float(tau2)
        self.pi = float(prior_sparsity)
        self.prior_mode = prior_mode
        self.a_pi = float(a_pi)
        self.b_pi = float(b_pi)
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # Store sufficient statistics
        self.reset()

    # -------- Core state --------
    def reset(self):
        """ Reset sufficient statistics to zero. """
        self.n = 0
        self.XtX = torch.zeros((self.d, self.d), dtype=self.dtype, device=self.device)
        self.Xty = torch.zeros((self.d, 1), dtype=self.dtype, device=self.device)
        self.yty = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # Enumerate all subsets as binary masks
        self.masks: List[Tuple[int, ...]] = [tuple(int(b) for b in np.binary_repr(m, width=self.d))
                                             for m in range(2 ** self.d)]
        self.log_post = torch.full((2 ** self.d,), -float("inf"), dtype=self.dtype, device=self.device)
        self.post = torch.full_like(self.log_post, fill_value=1.0 / (2 ** self.d))

    # -------- Data ingestion --------
    def add_datapoint(self, x, y):
        """
        Consume one observed (x, y), updating sufficient stats.
        x: (1, d) tensor (executed intervention's X values)
        y: scalar tensor (observed Y)
        """
        x = x.to(self.device, self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, d)
        y = y.to(self.device, self.dtype).reshape(-1, 1)  # (1,1)

        self.XtX += x.T @ x
        self.Xty += x.T @ y
        self.yty += (y.T @ y).squeeze()
        self.n += 1

    # -------- Posterior update --------
    def _log_prior_S(self, mask):
        k = sum(mask)
        if self.prior_mode == "bernoulli":
            return k*math.log(self.pi) + (self.d-k)*math.log(1.0 - self.pi)
        elif self.prior_mode == "betabinom":
            # Proportional part of Beta-Binomial pmf (constants cancel across S)
            a, b, d = self.a_pi, self.b_pi, self.d
            return (math.lgamma(k + a) + math.lgamma(d - k + b) - math.lgamma(d + a + b))
        else:
            return 0.0
    
    def _log_marginal_likelihood_from_stats(self, XtX, Xty, yty, n, mask):
        """
        Conjugate log p(y | X_S) for a given mask and provided sufficient stats.
        """
        idx = [i for i, z in enumerate(mask) if z == 1]
        p = len(idx)

        # Subset prior log p(S)
        logpS = self._log_prior_S(mask)

        if n == 0:
            # Uniform marginal if no data; log evidence cancels out across subsets, keep only prior on S
            return logpS

        if p == 0:
            # No parents: y ~ N(0, sigma^2); marginal integrates out beta trivially
            a_n = self.a0 + 0.5 * n
            b_n = self.b0 + 0.5 * yty.item()
            log_det_ratio = 0.0
            quad_term = 0.0
        else:
            XtX_S = XtX[idx][:, idx]  # (p,p)
            Xty_S = Xty[idx, :]  # (p,1)

            # beta | sigma^2 ~ N(0, sigma^2 tau^2 I) -> V0 = tau^2 I
            V0_inv = (1.0 / self.tau2) * torch.eye(p, dtype=self.dtype, device=self.device)
            Vn_inv = V0_inv + XtX_S

            # Cholesky for stability
            L = torch.linalg.cholesky(Vn_inv)
            logdet_Vn_inv = 2.0 * torch.sum(torch.log(torch.diag(L)))
            logdet_Vn = -logdet_Vn_inv

            logdet_V0 = p * math.log(self.tau2)
            log_det_ratio = logdet_V0 - logdet_Vn

            # (X^T y)^T Vn^{-1} (X^T y)
            u = torch.cholesky_solve(Xty_S, L)
            quad_term = (Xty_S.T @ u).item()

            a_n = self.a0 + 0.5 * n
            b_n = self.b0 + 0.5 * (yty.item() - quad_term)

        # log p(y|X_S) up to constants common across S
        log_ml = (
            0.5 * log_det_ratio
            + self.a0 * math.log(self.b0)
            - a_n * math.log(b_n)
            + torch.lgamma(torch.tensor(a_n, dtype=self.dtype)).item()
            - torch.lgamma(torch.tensor(self.a0, dtype=self.dtype)).item()
            - 0.5 * n * math.log(2.0 * math.pi)
        )

        return logpS + log_ml

    def _posterior_from_stats(self, XtX, Xty, yty, n):
        """
        Compute posterior over subsets from given sufficient stats (non-mutating).
        Returns posterior probs tensor shape (2^d,).
        """
        log_posts = []
        for mask in self.masks:
            lp = self._log_marginal_likelihood_from_stats(XtX, Xty, yty, n, mask)
            log_posts.append(lp)
        log_posts = torch.tensor(log_posts, dtype=self.dtype, device=self.device)

        m = torch.max(log_posts)
        probs = torch.exp(log_posts - m)
        probs = probs / probs.sum()
        return probs

    def update_posterior(self):
        """
        Update self.post/self.log_post from current sufficient stats.
        """
        self.post = self._posterior_from_stats(self.XtX, self.Xty, self.yty, self.n)
        # Keep a log copy for diagnostics
        with torch.no_grad():
            m = torch.max(torch.log(self.post + 1e-40))
            self.log_post = torch.log(self.post + 1e-40) - m + m  # Numerically safe

    # -------- Queries --------
    def edge_posterior(self):
        """
        Return vector of P(edge j present | data) by summing posterior mass over subsets with z_j = 1.
        """
        if self.post.numel() == 1:
            return torch.zeros(self.d, dtype=self.dtype, device=self.device) + 0.5

        pj = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        for mask, pS in zip(self.masks, self.post):
            for j, z in enumerate(mask):
                if z == 1:
                    pj[j] += pS
        return pj

    def most_probable_set(self):
        """
        Return MAP parent set (mask tuple) and its posterior probability.
        """
        idx = int(torch.argmax(self.post).item())
        return self.masks[idx], float(self.post[idx].item())
    
    # -------- "peek" (non-mutating) utilities for EEIG --------
    @torch.no_grad()
    def peek_update_edge_posterior(self, x_new, y_new):
        """
        Return edge posterior vector AFTER a virtual update with (x_new, y_new), without modifying internal state.
        """
        x = x_new.to(self.device, self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = y_new.to(self.device, self.dtype).reshape(-1, 1)

        XtX_new = self.XtX + x.T @ x
        Xty_new = self.Xty + x.T @ y
        yty_new = self.yty + (y.T @ y).squeeze()
        n_new = self.n + 1

        post_new = self._posterior_from_stats(XtX_new, Xty_new, yty_new, n_new)

        # Marginalize to edge probabilities
        pj = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        for mask, pS in zip(self.masks, post_new):
            for j, z in enumerate(mask):
                if z == 1:
                    pj[j] += pS
        return pj

    @torch.no_grad()
    def peek_update_edge_entropy(self, x_new, y_new):
        """
        Return sum_j H(edge_j | data & {(x_new,y_new)}), where H is Bernoulli entropy.
        Useful as an "expected posterior entropy" term for EEIG.
        """
        pj = self.peek_update_edge_posterior(x_new, y_new)
        eps = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        H = -(pj * torch.log(pj + eps) + (1 - pj) * torch.log(1 - pj + eps))
        return H.sum()
    

# ---- Convenience wrapper on top of ParentPosterior ----
class LocalParentPosterior(ParentPosterior):
    """
    Thin adapter that knows how to slice a full outcome vector into (x_parents, y_target)
    and provides safe add/peek methods that can skip updates when intervening on the target.
    """
    def __init__(self, parent_idx, target_idx, **kwargs):
        super().__init__(d=len(parent_idx), **kwargs)
        self.parent_idx = list(parent_idx)
        self.target_idx = int(target_idx)

    def slice_xy(self, outcome):
        out = outcome.to(self.device, self.dtype).view(-1)
        x = out[self.parent_idx].unsqueeze(0)  # (1, d)
        y = out[self.target_idx].view(1, 1)  # (1, 1)
        return x, y

    def add_datapoint_full(self, outcome, do_idx):
        # Only skip when we intervened on the target itself
        if do_idx == self.target_idx:
            return
        x, y = self.slice_xy(outcome)
        super().add_datapoint(x, y)  # update XtX/Xty/yty/n
    
    def num_datapoints(self) -> int:
        return int(self.n)

    
    # --------- Peeking helpers ---------
    @torch.no_grad()
    def peek_update_edge_posterior_full(self, outcome, intervened_idx=None):
        if intervened_idx == self.target_idx:
            return self.edge_posterior()
        x, y = self.slice_xy(outcome)
        return self.peek_update_edge_posterior(x, y)

    @torch.no_grad()
    def peek_update_edge_entropy_full(self, outcome, intervened_idx=None):
        if intervened_idx == self.target_idx:
            pj = self.edge_posterior()
            eps = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
            H = -(pj * torch.log(pj + eps) + (1 - pj) * torch.log(1 - pj + eps))
            return H.sum()
        x, y = self.slice_xy(outcome)
        return self.peek_update_edge_entropy(x, y)
    
    @torch.no_grad()
    def peek_update_edge_entropy_batch(self, X_batch, y_batch):
        """
        Vectorized version:
        X_batch: (N, d)  each row is one x_new
        y_batch: (N, 1)  each row is one y_new
        Returns:
        H_batch: (N,) tensor where H_batch[i] is sum_j H(edge_j | data & {(x_i,y_i)})
        """
        if X_batch.dim() != 2:
            raise ValueError("X_batch must be (N, d)")
        if y_batch.dim() != 2 or y_batch.size(1) != 1:
            raise ValueError("y_batch must be (N, 1)")
        N = X_batch.size(0)

        H_list = []
        for i in range(N):
            H_i = self.peek_update_edge_entropy(X_batch[i:i+1], y_batch[i:i+1])
            H_list.append(H_i)

        return torch.stack(H_list).view(-1)