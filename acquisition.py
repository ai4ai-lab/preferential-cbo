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
def eeig_pairwise(flow, ppost, x_anchor, candidates, s=1.0, *, 
                  anchor_xy=None, cand_xy=None, anchor_intervention=None, 
                  candidate_interventions=None):
    """
    Outcome-aware EEIG for pairwise comparisons.
    Requires (anchor_xy, cand_xy) providing true (x,y) for anchor and each candidate.
    Accounts for intervention blocking.
    Returns np.ndarray of shape (N,).
    """
    if anchor_xy is None or cand_xy is None:
        raise ValueError("eeig_pairwise requires anchor_xy and cand_xy")
    
    # 1) Prior edge entropy (considering intervention effects)
    if anchor_intervention is not None:
        # If we intervene on node i, edges into i provide no information
        prior_probs = ppost.edge_posterior()
        if anchor_intervention < len(prior_probs):
            prior_probs = prior_probs.clone()
            # Zero out intervened node's incoming edges
            for idx, parent_idx in enumerate(ppost.parent_idx):
                if parent_idx == anchor_intervention:
                    prior_probs[idx] = 0.5  # Maximum entropy
    else:
        prior_probs = ppost.edge_posterior()
    
    H_prior = binary_entropy(prior_probs).sum()
    
    # 2) Win probs from PrefFlow
    logf_a, _ = flow.f(x_anchor)
    logf_c, _ = flow.f(candidates)
    p_win = torch.sigmoid(((logf_a - logf_c) / s).flatten())
    
    # 3) Compute posterior entropies with intervention masking
    ig_vals = []
    for idx, ((cx, cy), pw) in enumerate(zip(cand_xy, p_win)):
        # Account for intervention blocking in posterior
        if candidate_interventions is not None and idx < len(candidate_interventions):
            cand_node = candidate_interventions[idx]
            if cand_node == ppost.target_idx:
                # Intervening on target blocks all learning
                H_cand = H_prior
            else:
                H_cand = ppost.peek_update_edge_entropy(cx, cy)
        else:
            H_cand = ppost.peek_update_edge_entropy(cx, cy)
        
        # Anchor branch
        ax, ay = anchor_xy
        H_anchor = ppost.peek_update_edge_entropy(ax, ay)
        
        # Weighted average
        H_post = pw * H_anchor + (1.0 - pw) * H_cand
        ig = torch.clamp(H_prior - H_post, min=0.0)
        ig_vals.append((ig / (H_prior + 1e-12)).item())
    
    return np.array(ig_vals)


@torch.no_grad()
def select_best_pair(flow, candidates, s=1.0):
    """
    Select best pair of interventions based on expected information.
    Returns indices of (anchor, opponent) that maximize information gain.
    """
    n = len(candidates)
    scores = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Compute expected entropy for this pair
            logf_i, _ = flow.f(candidates[i].unsqueeze(0))
            logf_j, _ = flow.f(candidates[j].unsqueeze(0))
            p_win = torch.sigmoid((logf_i - logf_j) / s).item()
            entropy = binary_entropy(torch.tensor(p_win))
            scores[i, j] = entropy.item()
    
    # Find best pair
    best_idx = np.unravel_index(np.argmax(scores), scores.shape)
    return best_idx[0], best_idx[1]