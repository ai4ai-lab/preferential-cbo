import torch
import numpy as np
from typing import Tuple, List
import math


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
        tau2_env: float = 10.0,
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
        self.tau2_env = float(tau2_env)

        self.d_env = 0
        self.reset()

    # -------- Core state --------
    def reset(self):
        """ Reset sufficient statistics to zero. """
        self.n = 0
        # stats are sized for (parents + env)
        D_tot = self.d + self.d_env
        self.XtX = torch.zeros((D_tot, D_tot), dtype=self.dtype, device=self.device)
        self.Xty = torch.zeros((D_tot, 1), dtype=self.dtype, device=self.device)
        self.yty = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # Enumerate only parent subsets
        self.masks: List[Tuple[int, ...]] = [tuple(int(b) for b in np.binary_repr(m, width=self.d))
                                            for m in range(2 ** self.d)]
        self.log_post = torch.full((2 ** self.d,), -float("inf"), dtype=self.dtype, device=self.device)
        self.post = torch.full_like(self.log_post, fill_value=1.0 / (2 ** self.d))

    # -------- Environment expansion --------
    def _expand_env(self, k: int):
        """Dynamically add k environment columns to the sufficient stats (non-destructive)."""
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
        # yty and n unchanged
        self.d_env = new_env

    # -------- Data ingestion --------
    def add_datapoint(self, x, y, x_env=None):
        """
        Consume one observed (x_parents, y) with optional environment one-hot x_env.
        x: (1, d)       parents
        x_env: (1, d_env_current) environment dummies (always included in the model)
        """
        x = x.to(self.device, self.dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = y.to(self.device, self.dtype).reshape(-1, 1)  # (1,1)

        if x_env is not None:
            x_env = x_env.to(self.device, self.dtype)
            if x_env.dim() == 1:
                x_env = x_env.unsqueeze(0)
            need_env = x_env.size(1)
            if need_env > self.d_env:
                self._expand_env(need_env - self.d_env)
            z = torch.cat([x, x_env], dim=1)  # (1, d + d_env)
        else:
            # assume stats currently match parent-only; if env exists, caller should pass it
            if self.d_env > 0 and x.size(1) == self.d:
                # pad implicit zeros for env (safe default)
                pad = torch.zeros((x.size(0), self.d_env), dtype=self.dtype, device=self.device)
                z = torch.cat([x, pad], dim=1)
            else:
                z = x

        self.XtX += z.T @ z
        self.Xty += z.T @ y
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
        Conjugate log p(y | X_S, env) where:
        - 'mask' selects a subset S of the first 'd' parent columns,
        - All environment columns (if present) are always included with prior variance tau2_env.
        XtX/Xty must be formed from the augmented design [parents | env].
        """
        # Prior over the selected parent set S
        logpS = self._log_prior_S(mask)

        if n == 0:
            return logpS

        # figure out how many env columns are in these stats
        D_tot = XtX.size(0)
        d_env = max(0, D_tot - self.d)

        # active indices: selected parents + all env columns
        idx_par = [i for i, z in enumerate(mask) if z == 1]
        idx_env = list(range(self.d, self.d + d_env))
        idx_all = idx_par + idx_env
        p = len(idx_all)

        if p == 0:
            # no parents, no env
            a_n = self.a0 + 0.5 * n
            b_n = self.b0 + 0.5 * yty.item()
            log_det_ratio = 0.0
            quad_term = 0.0
        else:
            idx = torch.tensor(idx_all, dtype=torch.long, device=self.device)
            XtX_S = XtX[idx][:, idx]
            Xty_S = Xty[idx, :]

            # diagonal prior precision: parents use tau2, env use tau2_env
            n_par = len(idx_par)
            n_env = len(idx_env)
            prior_prec = torch.full((p,), 1.0 / self.tau2, dtype=self.dtype, device=self.device)
            if n_env > 0:
                prior_prec[-n_env:] = 1.0 / self.tau2_env

            Vn_inv = XtX_S + torch.diag(prior_prec)

            L = torch.linalg.cholesky(Vn_inv)
            logdet_Vn_inv = 2.0 * torch.sum(torch.log(torch.diag(L)))
            logdet_Vn = -logdet_Vn_inv

            # log |V0| for mixed block
            logdet_V0 = n_par * math.log(self.tau2) + n_env * math.log(self.tau2_env)
            log_det_ratio = logdet_V0 - logdet_Vn

            u = torch.cholesky_solve(Xty_S, L)
            quad_term = (Xty_S.T @ u).item()

            a_n = self.a0 + 0.5 * n
            b_n = self.b0 + 0.5 * (yty.item() - quad_term)

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
    def peek_update_edge_posterior_aug(self, x_par, y, x_env=None):
        """
        Like edge_posterior() after a virtual update with (x_par, y, x_env), without mutating state.
        Handles env size larger than current by zero-padding a local copy of stats.
        """
        x_par = x_par.to(self.device, self.dtype)
        if x_par.dim() == 1:
            x_par = x_par.unsqueeze(0)
        y = y.to(self.device, self.dtype).reshape(-1, 1)

        # assemble z and a temporary stats buffer (pad if env is new)
        if x_env is not None:
            x_env = x_env.to(self.device, self.dtype)
            if x_env.dim() == 1:
                x_env = x_env.unsqueeze(0)
            need_env = x_env.size(1)
        else:
            need_env = self.d_env

        old_D = self.d + self.d_env
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
            if new_D > self.d:
                x_env = torch.zeros((1, new_D - self.d), dtype=self.dtype, device=self.device)
            else:
                x_env = torch.zeros((1, 0), dtype=self.dtype, device=self.device)

        z = torch.cat([x_par, x_env], dim=1)
        XtX_new = XtX_use + z.T @ z
        Xty_new = Xty_use + z.T @ y
        yty_new = self.yty + (y.T @ y).squeeze()
        n_new = self.n + 1

        post_new = self._posterior_from_stats(XtX_new, Xty_new, yty_new, n_new)

        pj = torch.zeros(self.d, dtype=self.dtype, device=self.device)
        for mask, pS in zip(self.masks, post_new):
            for j, zbit in enumerate(mask):
                if zbit == 1:
                    pj[j] += pS
        return pj

    @torch.no_grad()
    def peek_update_edge_posterior(self, x_new, y_new):
        # Backward-compatible wrapper: parent-only, no env
        return self.peek_update_edge_posterior_aug(x_new, y_new, x_env=None)

    @torch.no_grad()
    def peek_update_edge_entropy(self, x_new, y_new, x_env=None):
        pj = self.peek_update_edge_posterior_aug(x_new, y_new, x_env=x_env)
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
        self.env_map = {}  # global_node -> local env column index

    def slice_xy(self, outcome):
        out = outcome.to(self.device, self.dtype).view(-1)
        x = out[self.parent_idx].unsqueeze(0)  # (1, d)
        y = out[self.target_idx].view(1, 1)  # (1, 1)
        return x, y

    def add_datapoint_full(self, outcome, do_idx):
        if do_idx == self.target_idx:
            return
        x, y = self.slice_xy(outcome)
        x_env = self._x_env_onehot(int(do_idx))
        super().add_datapoint(x, y, x_env=x_env)
    
    def num_datapoints(self) -> int:
        return int(self.n)
    
    def _x_env_onehot(self, do_idx: int):
        """
        Return (1, d_env) one-hot for the intervened node (excluding the target).
        Dynamically expands parent stats with a new env column when a new do_idx appears.
        """
        if do_idx == self.target_idx:
            # env is "off" when we intervene on the target itself
            if self.d_env == 0:
                return torch.zeros((1, 0), dtype=self.dtype, device=self.device)
            return torch.zeros((1, self.d_env), dtype=self.dtype, device=self.device)

        if do_idx not in self.env_map:
            # allocate a fresh env column
            super()._expand_env(1)
            self.env_map[do_idx] = self.d_env - 1  # last index

        v = torch.zeros((1, self.d_env), dtype=self.dtype, device=self.device)
        v[0, self.env_map[do_idx]] = 1.0
        return v

    
    # --------- Peeking helpers ---------
    @torch.no_grad()
    def peek_update_edge_posterior_full(self, outcome, intervened_idx=None):
        if intervened_idx == self.target_idx:
            return self.edge_posterior()
        x, y = self.slice_xy(outcome)
        x_env = self._x_env_onehot(int(intervened_idx)) if intervened_idx is not None else None
        return self.peek_update_edge_posterior_aug(x, y, x_env=x_env)

    @torch.no_grad()
    def peek_update_edge_entropy_full(self, outcome, intervened_idx=None):
        if intervened_idx == self.target_idx:
            pj = self.edge_posterior()
            eps = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
            H = -(pj * torch.log(pj + eps) + (1 - pj) * torch.log(1 - pj + eps))
            return H.sum()
        x, y = self.slice_xy(outcome)
        x_env = self._x_env_onehot(int(intervened_idx)) if intervened_idx is not None else None
        return self.peek_update_edge_entropy(x, y, x_env=x_env)
    
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
    

# -------- Global helpers: build a DAG under acyclicity --------
@torch.no_grad()
def edge_matrix_from_locals(local_pp_list, d: int, device=None, dtype=torch.float64):
    """
    Build a (d x d) matrix P where P[j, i] ~ P(edge j -> i | data) from a list of LocalParentPosterior,
    one per target i. The LocalParentPosterior.d is the number of candidate parents for that i and
    its parent_idx gives which global nodes those are.

    Args:
        local_pp_list: list[LocalParentPosterior] of length = #targets you maintain
        d            : total number of variables in the system
    Returns:
        P: (d, d) tensor with P[j, i] = posterior prob of j -> i, zeros on diagonal
    """
    device = device or (local_pp_list[0].device if local_pp_list else torch.device("cpu"))
    dtype = dtype or (local_pp_list[0].dtype if local_pp_list else torch.float64)
    P = torch.zeros((d, d), dtype=dtype, device=device)

    for lpp in local_pp_list:
        # local edge posterior is length = len(parent_idx)
        pj_local = lpp.edge_posterior()  # probs for each local parent slot
        for slot, j_global in enumerate(lpp.parent_idx):
            P[j_global, lpp.target_idx] = pj_local[slot]
    # remove self-loops
    P.fill_diagonal_(0.0)
    return P


@torch.no_grad()
def _would_create_cycle(reach: torch.Tensor, u: int, v: int) -> bool:
    """
    reach[a, b] = True if a can reach b in the current graph.
    Adding u -> v would create a cycle iff reach[v, u] is already True (there's a path v -> ... -> u).
    """
    return bool(reach[v, u].item())


@torch.no_grad()
def greedy_map_dag_from_edge_matrix(
    P: torch.Tensor,
    score: str = "logit",  # "logit" or "prob"
    indegree_cap: int | None = None,
    forbid_self_loops: bool = True,
):
    """
    Project an edge score matrix P[j, i] to a DAG that maximizes the sum of selected edge scores,
    subject to acyclicity and optional per-node in-degree caps.

    Args:
        P            : (d, d) edge scores, e.g., posterior probabilities
        score        : "logit" (log-odds) or "prob" (raw probability) to rank edges
        indegree_cap : if not None, each node i can have at most this many incoming edges
        forbid_self_loops: ignore diagonal entries

    Returns:
        A: (d, d) binary adjacency matrix of a DAG (A[j, i] = 1 means j -> i)
    """
    d = P.size(0)
    device, dtype = P.device, P.dtype

    # Build candidate list
    edges = []
    for j in range(d):
        for i in range(d):
            if forbid_self_loops and i == j:
                continue
            p = float(P[j, i].item())
            if p <= 0.0:
                continue
            if score == "logit":
                # stable logit: clamp p away from {0,1}
                p_eps = min(max(p, 1e-12), 1 - 1e-12)
                w = math.log(p_eps) - math.log(1.0 - p_eps)
            elif score == "prob":
                w = p
            else:
                raise ValueError("score must be 'logit' or 'prob'")
            edges.append((w, j, i))

    # Sort by descending score
    edges.sort(key=lambda t: t[0], reverse=True)

    # Init graph and reachability
    A = torch.zeros((d, d), dtype=dtype, device=device)
    reach = torch.eye(d, dtype=torch.bool, device=device)  # reflexive reachability

    indeg = torch.zeros((d,), dtype=torch.int64, device=device)

    for w, u, v in edges:
        if indegree_cap is not None and int(indeg[v].item()) >= indegree_cap:
            continue
        if _would_create_cycle(reach, u, v):
            continue

        # Accept edge u -> v
        A[u, v] = 1.0
        indeg[v] += 1

        # Update reachability (transitive closure incremental update)
        # After adding u->v: any node x that reaches u can now reach all nodes that v reaches.
        # Update: reach[:, v] |= reach[:, u]; then for all z: if reach[v, z] True, reach[:, z] |= reach[:, u]
        # Implement this with vector ops:
        new_reach_to_v = reach[:, u]  # who reaches u
        # For all columns z with reach[v, z] == True, set reach[:, z] |= new_reach_to_v
        cols_to_update = reach[v, :].nonzero(as_tuple=False).view(-1)
        if cols_to_update.numel() > 0:
            reach[:, cols_to_update] = reach[:, cols_to_update] | new_reach_to_v.unsqueeze(1)
        # Also, v itself becomes reachable from those:
        reach[:, v] = reach[:, v] | new_reach_to_v

    return A