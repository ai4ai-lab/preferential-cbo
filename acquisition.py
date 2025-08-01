"""
Preference-Information-Gain (PIG) for k=2 (pairwise) comparisons.
For each candidate point x we sample a second point x' from a proposal
(Uniform on domain) and compute the expected entropy reduction
under the current PrefFlow and temperature s.

EIG(x) = H(post) - E_{y~p(y|x,x')}[ H(post|y) ]

We approximate with Monte-Carlo on the flow: one draw is enough in 2-D.
"""
import torch
import numpy as np

def pig_pairwise(flow, x, candidates, s):
    """
    x :  (1,d) tensor - the fixed first alternative
    candidates :  (N,d) tensor - N possible opponents
    returns :  (N,) numpy - estimated IG for each opponent
    """
    with torch.no_grad():
        # logf for both alternatives
        logf_x, _ = flow.f(x)  # (1,)
        logf_c, _ = flow.f(candidates)  # (N,)

        # probability x beats c  under current flow
        delta = (logf_x - logf_c) / s
        p_win = torch.sigmoid(delta)  # (N,)

        # Bernoulli entropy  H(p) = −p log p − (1−p)log(1−p)
        eps = 1e-9
        H_prior = -(p_win * torch.log(p_win+eps) +
                    (1-p_win) * torch.log(1-p_win+eps))  # (N,)

        # If we observe outcome y, posterior for that pair collapses.
        # Expected posterior entropy = p*0 + (1−p)*0 = 0
        # So IG = H_prior
        return H_prior.cpu().numpy()  # (N,)