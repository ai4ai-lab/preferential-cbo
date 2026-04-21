import torch
import torch.nn as nn
import math


class PrefGP(nn.Module):
    """
    Preferential Gaussian Process for preference learning.
    Drop-in replacement for PrefFlow in PCBO.
    
    Uses RBF kernel + Laplace approximation + Bradley-Terry likelihood.
    Provides the same interface as PrefFlow: f(X), logposterior(batch), s property.
    """
    
    def __init__(self, D, s=1.0, device=None, precision_double=False,
                 s_prior_mu=0.0, s_prior_sigma=0.25):
        super().__init__()
        self.D = D
        self.device = device or torch.device("cpu")
        self.precision_double = precision_double
        dtype = torch.float64 if precision_double else torch.float32
        
        # Kernel hyperparameters (learnable)
        self.log_lengthscale = nn.Parameter(torch.zeros(D, dtype=dtype, device=self.device))
        self.log_variance = nn.Parameter(torch.zeros(1, dtype=dtype, device=self.device))
        
        # Noise scale (same parameterization as PrefFlow)
        self.s_raw = nn.Parameter(torch.tensor(s, dtype=dtype, device=self.device).log())
        self.register_buffer("s_prior_mu", torch.tensor(s_prior_mu, dtype=dtype, device=self.device))
        self.register_buffer("s_prior_sigma", torch.tensor(s_prior_sigma, dtype=dtype, device=self.device))
        
        # GP state (set during training)
        self.X_train = None  # (M, D) inducing/training points
        self.f_map = None  # (M,) MAP function values
        self.K_inv = None  # (M, M) inverse kernel matrix (cached)
        
        self.ranking = False  # compatibility flag
    
    @property
    def s(self):
        dtype = torch.float64 if self.precision_double else torch.float32
        return torch.clamp(torch.exp(self.s_raw), min=torch.tensor(1e-6, dtype=dtype, device=self.device))
    
    def _kernel(self, X1, X2):
        """ARD RBF kernel."""
        ls = torch.exp(self.log_lengthscale)  # (D,)
        var = torch.exp(self.log_variance)  # (1,)
        # Scale inputs by lengthscale
        X1_s = X1 / ls.unsqueeze(0)  # (N1, D)
        X2_s = X2 / ls.unsqueeze(0)  # (N2, D)
        dist_sq = torch.cdist(X1_s, X2_s).pow(2)  # (N1, N2)
        return var * torch.exp(-0.5 * dist_sq)
    
    def _log_s_prior(self):
        """Same prior on s as PrefFlow."""
        log_s = self.s_raw
        mu, sigma = self.s_prior_mu, self.s_prior_sigma
        return (-0.5 * ((log_s - mu) / sigma) ** 2 
                - log_s - torch.log(sigma) 
                - 0.5 * torch.log(torch.tensor(2.0 * math.pi, device=self.device, dtype=log_s.dtype)))
    
    def _fit_laplace(self, X_all, pairs_w, pairs_l, n_newton=30):
        """
        Laplace approximation: find MAP of f given preferences,
        then cache K_inv for predictions.
        
        X_all: (M, D) unique training points
        pairs_w: (P,) indices into X_all for winners
        pairs_l: (P,) indices into X_all for losers
        """
        M = X_all.shape[0]
        dtype = X_all.dtype
        
        K = self._kernel(X_all, X_all) + 1e-5 * torch.eye(M, dtype=dtype, device=self.device)
        K_inv = torch.linalg.inv(K)
        
        # Newton's method for MAP estimate of f
        f = torch.zeros(M, dtype=dtype, device=self.device)
        
        for iteration in range(n_newton):
            # Preference likelihood gradient and Hessian
            diff = f[pairs_w] - f[pairs_l]  # (P,)
            p = torch.sigmoid(self.s * diff)  # P(winner beats loser)
            p = torch.clamp(p, 1e-7, 1 - 1e-7)
            
            # Gradient of log-likelihood
            grad_ll = torch.zeros(M, dtype=dtype, device=self.device)
            residual = self.s * (1 - p)  # (P,)
            grad_ll.scatter_add_(0, pairs_w, residual)
            grad_ll.scatter_add_(0, pairs_l, -residual)
            
            # Diagonal Hessian approximation of log-likelihood
            w = (self.s ** 2) * p * (1 - p)  # (P,)
            W_diag = torch.zeros(M, dtype=dtype, device=self.device)
            W_diag.scatter_add_(0, pairs_w, w)
            W_diag.scatter_add_(0, pairs_l, w)
            W_diag = torch.clamp(W_diag, min=1e-8)
            
            # Full gradient: grad_ll - K_inv @ f
            grad = grad_ll - K_inv @ f
            
            # Hessian: -diag(W) - K_inv
            # Newton step: f_new = f - H^{-1} @ grad
            H_inv_diag = -1.0 / (W_diag + torch.diag(K_inv))  # diagonal approx
            step = H_inv_diag * grad
            
            # Damped step for stability
            f = f + 0.5 * step
        
        self.X_train = X_all.detach()
        self.f_map = f.detach()
        self.K_inv = K_inv.detach()
    
    def f(self, X):
        """
        Predict utility at new points X. 
        Returns (f_pred, zeros) to match PrefFlow interface.
        """
        if self.X_train is None or self.f_map is None:
            # Not trained yet — return zeros
            return torch.zeros(X.shape[0], device=self.device), torch.zeros(X.shape[0], device=self.device)
        
        dtype = torch.float64 if self.precision_double else torch.float32
        X = X.to(dtype=dtype, device=self.device)
        
        K_star = self._kernel(X, self.X_train)  # (N, M)
        f_pred = K_star @ (self.K_inv @ self.f_map)
        
        return f_pred, torch.zeros_like(f_pred)
    
    def logposterior(self, batch, weightprior=1.0):
        """
        Compute log-posterior for training (matches PrefFlow interface).
        Also runs Laplace fitting internally.
        
        batch: (X_raw, Y) where X_raw is (2, D, N), Y is (N,) bool
        """
        X_raw, Y = batch
        N = Y.shape[0]
        dtype = torch.float64 if self.precision_double else torch.float32
        
        # Convert to pairs format: (2, D, N) -> winner/loser features
        X_pairs = X_raw.permute(2, 0, 1).reshape(-1, self.D).to(dtype)  # (2N, D)
        
        # Find unique points
        X_all, inverse = torch.unique(X_pairs, dim=0, return_inverse=True)
        
        # Build winner/loser index pairs
        base = torch.arange(N, device=self.device)
        w_raw = base * 2 + (~Y.bool()).long()  # index of loser in X_pairs (swapped)
        l_raw = base * 2 + Y.bool().long()
        # Actually: if Y=True, winner is index 2*i, loser is 2*i+1
        w_raw = torch.where(Y.bool(), base * 2, base * 2 + 1)
        l_raw = torch.where(Y.bool(), base * 2 + 1, base * 2)
        
        pairs_w = inverse[w_raw]
        pairs_l = inverse[l_raw]
        
        # Fit Laplace (updates self.X_train, self.f_map, self.K_inv)
        self._fit_laplace(X_all, pairs_w, pairs_l)
        
        # Compute log-likelihood at MAP
        diff = self.f_map[pairs_w] - self.f_map[pairs_l]
        log_p = torch.log(torch.sigmoid(self.s * diff) + 1e-12)
        loglik = log_p.sum()
        
        # GP prior term
        logprior_f = -0.5 * self.f_map @ self.K_inv @ self.f_map
        
        # Prior on s
        logprior_s = self._log_s_prior()
        
        return loglik + weightprior * logprior_f + logprior_s
    
    # Stable sampling/log_prob stubs (for compatibility)
    def sample_stable(self, num_samples=1, **kwargs):
        # GP doesn't sample in the same way; return from prior
        X = torch.randn(num_samples, self.D, device=self.device)
        lp = torch.zeros(num_samples, device=self.device)
        return X, lp
    
    def log_prob_stable(self, x, **kwargs):
        f_pred, _ = self.f(x)
        return f_pred