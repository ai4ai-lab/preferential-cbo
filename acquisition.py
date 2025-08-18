import torch
import numpy as np


@torch.no_grad()
def binary_entropy(p, eps=1e-12):
    """Elementwise Bernoulli entropy H(p) = -p log p - (1-p) log(1-p)"""
    p = torch.clamp(p, 0.0 + eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


@torch.no_grad()
def pig_pairwise(flow, x, candidates, s):
    """
    Preference-Information-Gain (PIG) for k=2 (pairwise) comparisons.
    For each candidate x, sample a second point x' from a proposal (Uniform on domain) 
    and compute the expected entropy reduction under the current PrefFlow and temperature s.

    Args:
    - flow: trained PrefFlow model
    - x: (1,d) tensor - the fixed first alternative
    - candidates: (N,d) tensor - N possible opponents
    - s: temperature parameter for softmax likelihood

    Returns:
    - entropy: (N,) numpy array - expected entropy reduction for each opponent
    """
    logf_x, _ = flow.f(x)  # (1,)
    logf_c, _ = flow.f(candidates)  # (N,)

    # Probability x beats c under current flow (logistic on logf difference)
    delta = (logf_x - logf_c) / s
    p_win = torch.sigmoid(delta).flatten()  # (N,)

    # Bernoulli entropy
    entropy = binary_entropy(p_win)
    return entropy.cpu().numpy()


@torch.no_grad()
def eeig_pairwise(flow, ppost, x_anchor, candidates, s=1.0, *, anchor_xy=None, cand_xy=None):
    """
    Outcome-aware EEIG for pairwise comparisons.
    Requires (anchor_xy, cand_xy) providing true (x,y) for anchor and each candidate.
    Returns np.ndarray of shape (N,).
    """
    if anchor_xy is None or cand_xy is None:
        raise ValueError("eeig_pairwise requires anchor_xy=(ax,ay) and cand_xy=[(cx,cy), ...].")

    # 1) Prior edge entropy
    H_prior = binary_entropy(ppost.edge_posterior()).sum()

    # 2) Win probs from PrefFlow
    logf_a, _ = flow.f(x_anchor)  # (1,)
    logf_c, _ = flow.f(candidates)  # (N,)
    p_win = torch.sigmoid(((logf_a - logf_c) / s).flatten())  # (N,)

    # 3) Anchor branch
    ax, ay = anchor_xy  # shapes (1, d_x), (1, 1)
    H_anchor = ppost.peek_update_edge_entropy(ax, ay)  # scalar tensor

    # 4) Candidate branches (loop)
    # Can switch peek_update_edge_entropy to batch version if needed
    ig_vals = []
    for (cx, cy), pw in zip(cand_xy, p_win):
        H_cand = ppost.peek_update_edge_entropy(cx, cy)
        H_post = pw * H_anchor + (1.0 - pw) * H_cand
        ig = torch.clamp(H_prior - H_post, min=0.0)
        ig_vals.append((ig / (H_prior + 1e-12)).item())

    return np.array(ig_vals)