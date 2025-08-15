import torch
import numpy as np


@torch.no_grad()
def binary_entropy(p, eps=1e-9):
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
def eeig_pairwise(flow, ppost, x_anchor, candidates, s=1.0, anchor_xy=None, cand_xy=None):
    """
    Edge-Entropy Information Gain (EEIG) for pairwise comparisons.

    If (anchor_xy, cand_xy) are provided, we compute outcome-aware EEIG by
    peeking the causal posterior with the true (x,y) of each hypothetical outcome.
    Otherwise, we fall back to the feature-only approximation with y=0.

    Returns:
        np.ndarray shape (N,) with expected reduction in sum of edge entropies.
    """
    # 1) Prior edge entropy
    H_prior = binary_entropy(ppost.edge_posterior()).sum()

    # 2) Preference win probs from PrefFlow
    logf_anchor, _ = flow.f(x_anchor)  # (1,)
    logf_cand, _ = flow.f(candidates)  # (N,)
    p_win = torch.sigmoid(((logf_anchor - logf_cand) / s).flatten())  # (N,)

    # 3) Decide which (x,y) to use for the virtual updates
    use_true_outcomes = (anchor_xy is not None) and (cand_xy is not None)

    # Prepare anchor/candidate (x,y)
    if use_true_outcomes:
        ax, ay = anchor_xy
        ax = ax if ax.dim() == 2 else ax.unsqueeze(0)   # (1, d)
        ay = ay.reshape(1, 1) if torch.is_tensor(ay) else torch.tensor([[float(ay)]], dtype=ppost.dtype, device=ppost.device)
    else:
        ax = x_anchor
        ay = torch.zeros((1, 1), dtype=ppost.dtype, device=ppost.device)  # dummy y

    ig_vals = []

    for i in range(candidates.shape[0]):
        if use_true_outcomes:
            cx, cy = cand_xy[i]
            cx = cx if cx.dim() == 2 else cx.unsqueeze(0)
            cy = cy.reshape(1, 1) if torch.is_tensor(cy) else torch.tensor([[float(cy)]], dtype=ppost.dtype, device=ppost.device)
        else:
            cx = candidates[i:i+1]
            cy = torch.zeros((1, 1), dtype=ppost.dtype, device=ppost.device)

        # 4) Peek posterior edge entropies for each outcome branch:
        # - anchor wins -> add (ax, ay)
        # - candidate wins -> add (cx, cy)
        H_anchor = ppost.peek_update_edge_entropy(ax, ay)
        H_cand = ppost.peek_update_edge_entropy(cx, cy)

        # 5) Expected posterior entropy under PrefFlow's outcome probabilities
        H_post = p_win[i] * H_anchor + (1.0 - p_win[i]) * H_cand

        ig_vals.append((H_prior - H_post).item())

    return np.array(ig_vals)