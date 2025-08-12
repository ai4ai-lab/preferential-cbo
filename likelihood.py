import torch
from itertools import combinations


def elementary_symmetric_sum(fvalues, l):
    """l-th elementary symmetric sum over a 1D tensor of values"""
    n = fvalues.numel()
    if l < 0 or l > n:
        return torch.zeros((), device=fvalues.device, dtype=fvalues.dtype)
    if l == 0:
        return torch.ones((), device=fvalues.device, dtype=fvalues.dtype)  # Return 1 for l=0 (sum of products of zero elements)
    if l == 1:
        return fvalues.sum()
    
    # Combinatorial, only suitable for small n
    idx = list(combinations(range(n), l))
    if not idx:
        return torch.zeros((), device=fvalues.device, dtype=fvalues.dtype)
    idx = torch.tensor(idx, device=fvalues.device, dtype=torch.long)
    # Gather rows, take product along last dim, sum over all combinations
    return torch.prod(fvalues[idx], dim=1).sum()


def exp_rum_pairwise_prob(f_i, f_j, s):
    """
    Exponential RUM pairwise probability P(i beats j).

    Uses stable closed form: let z = f_i - f_j, then
    P = 0.5 - 0.5 * sign(z) * expm1(-s * |z|)
    """
    z = f_i - f_j
    return 0.5 - 0.5 * z.sign() * torch.expm1(-s * z.abs())


def exp_rum_likelihood(f_x, f_others, s, k):
    """
    Exponential RUM likelihood that x is the k-wise winner among x and others.
    Works for any k >= 2. Has a fast closed-form path when k = 2.

    Args:
    - f_x: scalar tensor, utility of the candidate x
    - f_others: tensor shape (k-1,), utilities of the other items
    - s: positive scalar tensor (noise precision)
    - k: total set size (must equal f_others.numel() + 1)

    Returns:
    - scalar tensor: P(x wins | f_x, f_others, s)
    """
    # Fast path for pairwise
    if k == 2:
        p = exp_rum_pairwise_prob(f_x, f_others[0], s)
        return torch.clamp(p, 1e-12, 1.0 - 1e-12)
    
    # General case for k > 2 (Mikkola et al., RUM with Exp noise)
    prob = torch.zeros((), device=f_x.device, dtype=f_x.dtype)
    f_star = f_others.max()

    # Precompute terms used in symmetric sums
    diff_terms = -torch.exp(-s * (f_x - f_others))

    for l in range(0, k):
        # exp(-s * (l+1) * max(f_star - f_x, 0))
        exp_term = torch.exp(-s * (l + 1) * (torch.clamp(f_star - f_x, min=0)))
        factor = 1.0 / (l + 1)
        sym_sum = elementary_symmetric_sum(diff_terms, l)
        prob += factor * exp_term * sym_sum

    # Clamp for numerical stability
    prob = torch.clamp(prob, 1e-12, 1.0 - 1e-12)
    return prob