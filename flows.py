import torch
import normflows as nf


# -------- Base Distribution --------
def create_base_distribution(D, device, precision_double):
    """
    Diagonal Gaussian base distribution for the flow.
    """
    base = nf.distributions.base.DiagGaussian(D)
    base = base.to(device)
    return base.double() if precision_double else base.float()


# -------- Normalizing Flows --------
def real_nvp(K, D, q0, device, precision_double, hidden_multiplier=2):
    """
    Build a RealNVP-style masked affine coupling flow with ActNorm after each block.

    Args:
    - K (int): Number of flow blocks.
    - D (int): Dimensionality of the latent space.
    - q0 (torch.Tensor): Initial base distribution parameters.
    - device (torch.device): Device to place the flow on.
    - precision_double (bool): Whether to use double precision.
    """
    # Alternating binary mask [1, 0, 1, 0, ...]
    mask_vec = torch.tensor([1 if i % 2 == 0 else 0 for i in range(D)]).to(device)
    mask_vec = mask_vec.double() if precision_double else mask_vec.float()

    flows = []
    hidden_dim = D * hidden_multiplier

    for i in range(K):
        s_net = nf.nets.MLP([D, hidden_dim, D], init_zeros=True)
        t_net = nf.nets.MLP([D, hidden_dim, D], init_zeros=True)

        # Alternate masks for each flow block
        mask = mask_vec if (i % 2 == 0) else (1 - mask_vec)

        flows.append(nf.flows.MaskedAffineFlow(mask, t_net, s_net))
        flows.append(nf.flows.ActNorm(D))

    # Precision: cast whole flow graph
    if not precision_double:
        flows = [f.float() for f in flows]
        q0 = q0.float()

    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    nfm = nfm.to(device)
    nfm = nfm.float() if not precision_double else nfm.double()

    # ActNorm warmup: sample to initialize ActNorm parameters
    nfm.eval()
    _ = nfm.sample(num_samples=2 ** (D + 5))
    nfm.train()

    return nfm


def residual_flow(K, D, q0, device, precision_double, hidden_units=50, hidden_layers=3):
    """
    Build a Residual Flow (Neural ODE–style blocks with Lipschitz MLP) + ActNorm.

    Args:
    - K (int): Number of flow blocks.
    - D (int): Dimensionality of the latent space.
    - q0 (torch.Tensor): Initial base distribution parameters.
    - device (torch.device): Device to place the flow on.
    - precision_double (bool): Whether to use double precision.
    """
    flows = []
    for _ in range(K):
        net = nf.nets.LipschitzMLP(
            [D] + [hidden_units] * (hidden_layers - 1) + [D],
            init_zeros=True, 
            lipschitz_const=0.9
        )
        flows.append(nf.flows.Residual(net, reduce_memory=True))
        flows.append(nf.flows.ActNorm(D))

    # Precision: cast whole flow graph
    if not precision_double:
        flows = [flow.float() for flow in flows]
        q0 = q0.float()
    
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    nfm = nfm.to(device)
    nfm = nfm.float() if not precision_double else nfm.double()

    # ActNorm warmup: sample to initialize ActNorm parameters
    nfm.eval()
    _ = nfm.sample(num_samples=2 ** (D + 5))
    nfm.train()

    return nfm

def neural_spline_flow(K, D, q0, device, precision_double, hidden_layers=2, hidden_units=128):
    """
    Build a Neural Spline Flow with Autoregressive Rational Quadratic Spline and LULinearPermute.
    
    Args:
    - K (int): Number of flow blocks.
    - D (int): Dimensionality of the latent space.
    - q0 (torch.Tensor): Initial base distribution parameters.
    - device (torch.device): Device to place the flow on.
    - precision_double (bool): Whether to use double precision.
    """
    flows = []
    for _ in range(K):
        flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(D, hidden_layers, hidden_units))
        flows.append(nf.flows.LULinearPermute(D))

    # Precision: cast whole flow graph
    if not precision_double:
        flows = [flow.float() for flow in flows]
        q0 = q0.float()
    
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    nfm = nfm.to(device)
    nfm = nfm.float() if not precision_double else nfm.double()

    # ActNorm warmup: sample to initialize ActNorm parameters
    nfm.eval()
    _ = nfm.sample(num_samples=2 ** (D + 5))
    nfm.train()

    return nfm