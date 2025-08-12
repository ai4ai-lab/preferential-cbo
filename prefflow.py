import torch
from normflows.core import NormalizingFlow
from likelihood import exp_rum_likelihood, exp_rum_pairwise_prob


class PrefFlow(NormalizingFlow):
    """
    Normalizing flow for preference learning.
    
    - Primary path: pairwise comparisons (fast, stable).
    - Also supports k-wise ranking if ranking=True and inputs follow the expected k-wise shape.

    Expected shapes
    ----------------
    Pairwise:
      X_raw: (2, D, N)  # N pairs; first row is item A, second row is item B
      Y: (N,) bool  # True if first item wins, False if second wins

    K-wise ranking:
      X: (k, D, n)  # For each of n rankings, a list of k items ordered by rank
                    # (row 0 is the winner, row 1 second, ... row k-1 last)
    """

    def __init__(self, nflow, s, D, ranking, device, precision_double, s_prior_mu=0.0, s_prior_sigma=0.25):
        super().__init__(nflow.q0, nflow.flows)

        self.ranking = ranking
        self.D = D
        self.device = device
        self.precision_double = precision_double
        self.flowname = nflow.__class__.__name__

        dtype = torch.float64 if precision_double else torch.float32
        self.register_buffer("s_prior_mu", torch.tensor(s_prior_mu, dtype=dtype, device=device))
        self.register_buffer("s_prior_sigma", torch.tensor(s_prior_sigma, dtype=dtype, device=device))

        # learnable log s
        self.register_parameter("s_raw", torch.nn.Parameter(torch.tensor(s, dtype=dtype, device=device).log()))

    # -------- stable wrappers
    def sample_stable(self, num_samples=1, max_batch_size=1000):
        """Sample in chunks to avoid out of memory issues; returns (samples, logprobs)"""
        remaining = int(num_samples)
        samples, logprobs = [], []
        while remaining > 0:
            b = min(max_batch_size, remaining)
            s, lp = super().sample(b)
            samples.append(s.detach())
            logprobs.append(lp.detach())
            remaining -= b
        X = torch.cat(samples, 0)
        LP = torch.cat(logprobs, 0)
        # filter any pathological NaNs/Infs
        mask = ~(torch.isnan(X).any(dim=1) | torch.isinf(X).any(dim=1))
        X = X[mask]
        LP = LP[mask[: LP.shape[0]]]
        return X, LP
    
    def log_prob_stable(self, x, max_batch_size=1000):
        """Log prob in chunks to avoid out of memory issues; returns (N,) tensor."""
        n = x.shape[0]
        out = []
        start = 0
        dtype = torch.float64 if self.precision_double else torch.float32
        while start < n:
            end = min(start + max_batch_size, n)
            lp = super().log_prob(x[start:end, :].to(dtype))
            out.append(lp.detach())
            start = end
        LP = torch.cat(out, 0)
        return LP[~torch.isnan(LP)]

    # -------- model internals
    @property
    def s(self):
        """Positive noise precision; tiny floor for stability."""
        dtype = torch.float64 if self.precision_double else torch.float32
        return torch.clamp(torch.exp(self.s_raw), min=torch.tensor(1e-6, dtype=dtype, device=self.device))

    def _pairs_to_matrix(self, X_raw):
        """
        Vectorized: (2, D, N) -> (2N, D)
        Preserves pair order: rows [2*i] and [2*i+1] are a pair.
        """
        # (2, D, N) -> (N, 2, D) -> (2N, D)
        return X_raw.permute(2, 0, 1).reshape(-1, self.D)

    def _select_winners(self, logf_pairs, X_pairs, Y_bool):
        """
        Vectorized winner selection for pairwise:
            logf_pairs: (2N,)
            X_pairs: (2N, D)
            Y_bool: (N,) True -> pick first; False -> pick second

        Returns:
            winners_logf: (N,)
            winners_X: (N, D)
        """
        N = Y_bool.shape[0]
        base = torch.arange(N, device=self.device) * 2
        # if Y=False pick offset +1
        idx = base + (~Y_bool).long()

        return logf_pairs[idx], X_pairs[idx, :]

    def f(self, X):
        """
        Given points X, return (logf, logdetJinv), both (N,) tensors.
        logf = log p_flow(X) = log q0(u) + log |det J^{-1}(X)|
        """
        u, logdetJinv = self.inverse_and_log_det(X)
        logf = self.q0.log_prob(u) + logdetJinv

        # Handle NaNs
        logf[torch.isnan(logf)] = float('-inf')
        logdetJinv[torch.isnan(logdetJinv)] = float('-inf')
        return logf, logdetJinv
    
    def _log_s_prior(self):
        """
        Log-density of LogNormal(s; mu, sigma) with parameterization log s ~ N(mu, sigma^2):
        log p(s) = -0.5 * ((log s - mu)/sigma)^2 - log s - log(sigma*sqrt(2*pi))
        Returned as a scalar tensor.
        """
        log_s = self.s_raw
        mu, sigma = self.s_prior_mu, self.s_prior_sigma
        return -0.5 * ((log_s - mu) / sigma) ** 2 - log_s - torch.log(sigma) - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=log_s.device, dtype=log_s.dtype))

    def logposterior(self, batch, weightprior=1.0):
        """
        FS-MAP objective:
            log p(prefs | f, s) + weightprior * sum_{winners} f(x_winner) + log p(s)

        Pairwise batch: (X_raw, Y)
          - X_raw: (2, D, N)
          - Y: (N,) bool

        K-wise ranking batch: X of shape (k, D, n), where row 0 is winner per ranking
        """
        if not self.ranking:
            # Pairwise comparisons
            X_raw, Y = batch
            X_pairs = self._pairs_to_matrix(X_raw)  # (2N, D)
            logf_pairs, _ = self.f(X_pairs)  # (2N,)

            # Likelihood
            f_a = logf_pairs[::2]
            f_b = logf_pairs[1::2]
            prob = torch.where(Y.bool(), 
                               exp_rum_pairwise_prob(f_a, f_b, self.s),
                               exp_rum_pairwise_prob(f_b, f_a, self.s))
            prob = torch.clamp(prob, min=1e-12, max=1.0 - 1e-12)
            loglik = torch.log(prob).sum()

            # Winner prior term (sum of f at winners)
            winners_logf, _ = self._select_winners(logf_pairs, X_pairs, Y.bool())
            logprior_f = winners_logf.sum()

        else:
            # k-wise ranking: X of shape (k, D, n), winner is row 0 for each column
            X = batch
            k, d, n = X.shape
            # Reshape to (k*n, d) for flow
            X_flat = X.permute(2, 0, 1).reshape(k * n, d)
            logf_flat, _ = self.f(X_flat)
            logf = logf_flat.view(k, n).transpose(0, 1)  # (k, n)
            
            # Likelihood: product of sequential winner selections
            loglik = torch.zeros((), device=self.device, dtype=logf.dtype)
            for i in range(n):
                for j in range(k - 1):
                    f_x = logf[j, i]
                    f_others = logf[j + 1:, i]
                    loglik = loglik + torch.log(exp_rum_likelihood(f_x, f_others, self.s, k))

            # Winner prior term
            winners_logf = logf[0, :]
            logprior_f = winners_logf.sum()

        # Prior on s
        logprior_s = self._log_s_prior()

        # FS-MAP objective
        return loglik + weightprior * logprior_f + logprior_s