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
        d: int,
        a0: float = 1.0,  # Inverse Gamma prior shape
        b0: float = 1.0,  # Inverse Gamma prior scale
        tau2: float = 1.0,  # prior scale for beta (ridge)
        prior_sparsity: float = 0.5,  # Bernoulli(pi) per edge
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.d = d
        self.a0 = float(a0)
        self.b0 = float(b0)
        self.tau2 = float(tau2)
        self.pi = float(prior_sparsity)
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.reset()

    # -------- Core state --------
    def reset(self):
        self.n = 0
        self.XtX = torch.zeros((self.d, self.d), dtype=self.dtype, device=self.device)
        self.Xty = torch.zeros((self.d, 1), dtype=self.dtype, device=self.device)
        self.yty = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # enumerate all subsets as binary masks
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
    def _log_marginal_likelihood_from_stats(self, XtX, Xty, yty, n, mask):
        """
        Conjugate log p(y | X_S) for a given mask and provided sufficient stats.
        """
        idx = [i for i, z in enumerate(mask) if z == 1]
        p = len(idx)

        # subset prior log p(S)
        logpS = sum((math.log(self.pi) if z == 1 else math.log(1.0 - self.pi)) for z in mask)

        if n == 0:
            # uniform marginal if no data; log evidence cancels out across subsets, keep only prior on S
            return logpS

        if p == 0:
            # No parents: y ~ N(0, σ^2); marginal integrates out β trivially
            a_n = self.a0 + 0.5 * n
            b_n = self.b0 + 0.5 * yty.item()
            log_det_ratio = 0.0
            quad_term = 0.0
        else:
            XtX_S = XtX[idx][:, idx]   # (p,p)
            Xty_S = Xty[idx, :]        # (p,1)

            # β | σ^2 ~ N(0, σ^2 τ^2 I) -> V0 = τ^2 I
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
        # keep a log copy for diagnostics
        with torch.no_grad():
            m = torch.max(torch.log(self.post + 1e-40))
            self.log_post = torch.log(self.post + 1e-40) - m + m  # numerically safe

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
    def peek_update_edge_posterior(self, x_new: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
        """
        Return edge posterior vector AFTER a virtual update with (x_new, y_new),
        without modifying internal state.
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

        # marginalize to edge probabilities
        pj = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        for mask, pS in zip(self.masks, post_new):
            for j, z in enumerate(mask):
                if z == 1:
                    pj[j] += pS
        return pj

    @torch.no_grad()
    def peek_update_edge_entropy(self, x_new: torch.Tensor, y_new: torch.Tensor) -> torch.Tensor:
        """
        Return sum_j H( edge_j | data ∪ {(x_new,y_new)} ), where H is Bernoulli entropy.
        Useful as an "expected posterior entropy" term for EEIG.
        """
        pj = self.peek_update_edge_posterior(x_new, y_new)
        eps = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        H = -(pj * torch.log(pj + eps) + (1 - pj) * torch.log(1 - pj + eps))
        return H.sum()