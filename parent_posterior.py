"""
Maintain and update a Bayesian posterior over the direct parents of Y.
Linear SCM case (closed-form); GP/RFF can be swapped in later.
"""
import itertools
import torch
import numpy as np

class ParentPosterior:
    def __init__(self, d, sigma_eps=0.1, sigma_theta=1.0, prior_sparsity=0.5):
        """
        d : # candidate covariates (X1, ..., X_d) (Y is excluded)
        We enumerate all 2^d parent sets for clarity (ok for small d).
        When scaling to > 10, we use a more efficient enumeration with MCMC or variational approximation.
        sigma_eps : noise on Y (observation)
        sigma_theta : prior variance on theta (linear-regression prior)
        prior_sparsity : Bernoulli prior of a covariate being a parent (for each subset)
        """
        self.d = d
        # enumerate all parent sets
        self.S = list(itertools.product([0,1], repeat=d))  # binary mask per subset, length 2^d
        self.M = len(self.S)
        # log prior P(G) -> log P(G) = log P(sparsity)^|S| * log(1-P(sparsity))^(d-|S|)
        logp0 = np.array([np.sum(s)*np.log(prior_sparsity) +
                          (d-np.sum(s))*np.log(1-prior_sparsity)
                          for s in self.S])
        self.log_post = torch.tensor(logp0, dtype=torch.float64)

        # linear-regression hyper‐params
        self.sigma_eps = sigma_eps
        self.sigma_theta = sigma_theta

        # store sufficient statistics (X^T X, X^T y), needed for marginal likelihood (Bayesian linear regression)
        self.XTX = torch.zeros((d,d), dtype=torch.float64)
        self.XTy = torch.zeros(d, dtype=torch.float64)
        self.n = 0

    # update with (x, y) observational or interventional (e.g. winner of preference query)
    def add_datapoint(self, x, y):
        """
        x : 1xd tensor  (covariates)
        y : scalar
        """
        x = x.double()
        self.XTX += x.T @ x  # outer product, (dxd)
        self.XTy += (x.flatten()*y)  # inner product, (dx1)
        self.n += 1 

    # recompute log-marginal for every parent set
    # 1) select sub-matrix XTX and vector XTy by mask
    # 2) compute posterior covariance and mean
    # 3) compute log marginal likelihood
    def _log_marginal_likelihood(self):
        logml = torch.empty(self.M, dtype=torch.float64)
        for idx, s in enumerate(self.S):
            mask = torch.tensor(s, dtype=torch.bool)
            p = int(mask.sum())
            if p == 0:  # empty parent set
                # No parents: marginal likelihood reduces to constant (prob of observing y with only noise)
                logml[idx] = -self.n*np.log(self.sigma_eps)  # log P(Y) = -n log sigma_eps
            else:
                XTXs = self.XTX[mask][:,mask]
                XTy = self.XTy[mask]
                # Bayesian linear reg closed form
                # log P(Y | X_s) = -n/2 * log(2*pi*sigma_eps^2) + 0.5 * log det(Sigma_post) / det(Sigma0_inv) + 0.5 * (XTy @ mu_post) / sigma_eps^2
                Sigma0_inv = (1/self.sigma_theta**2)*torch.eye(p, dtype=torch.float64)
                Sigma_post = torch.inverse(Sigma0_inv + XTXs/self.sigma_eps**2)
                mu_post = Sigma_post @ (XTy/self.sigma_eps**2)
                quad = (XTy @ mu_post)
                log_det = torch.logdet(Sigma_post) - torch.logdet(Sigma0_inv)
                logml[idx] = (-self.n/2 * np.log(2*np.pi*self.sigma_eps**2)
                    + 0.5*log_det
                    + 0.5*quad/(self.sigma_eps**2)
                )
        return logml

    def update_posterior(self):
        logml = self._log_marginal_likelihood()
        self.log_post = self.log_post + logml  # add log-likelihood (Bayesian update)
        # normalise in log-space (avoid numerical issues)
        self.log_post -= torch.logsumexp(self.log_post, dim=0)

    # ---------- diagnostics ----------
    def edge_posterior(self):
        """
        Return a d-vector of P(X_j -> Y) by marginalising over sets.
        """
        post = torch.exp(self.log_post)
        probs = torch.zeros(self.d, dtype=torch.float64)
        for idx, s in enumerate(self.S):
            mask = torch.tensor(s, dtype=torch.float64)
            probs += post[idx] * mask
        return probs

    def most_probable_set(self):
        """
        Return the most probable parent set and its posterior probability.
        """
        idx = torch.argmax(self.log_post)
        return self.S[int(idx)], torch.exp(self.log_post[idx]).item()