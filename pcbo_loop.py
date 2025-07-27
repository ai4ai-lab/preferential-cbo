'''Automated experiment runner for the PCBO loop, to use once experiment is stable and ready.'''
from pcbo_dataset import PCBO_SyntheticDataset
from prefflow import PrefFlow
from parent_posterior import ParentPosterior
from dag_visual import plot_dag
from torch.utils.data import DataLoader
import torch, numpy as np, matplotlib.pyplot as plt
import normflows as nf
from flows import NeuralSplineFlow  # or RealNVP

# data
dataset = PCBO_SyntheticDataset(n_queries=0, k=2, d=2)  # start with no prefs, we request them on the fly
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # now unused, useful for batched updates later

# flow model
D = 2
q0 = nf.distributions.DiagGaussian(D, trainable=False)
nflows = 5
nfm = NeuralSplineFlow(nflows, D, q0, device='cpu', PRECISION_DOUBLE=False)
flow = PrefFlow(nfm, s=1.0, D=D, ranking=False, device='cpu', precision_double=False)
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-4)
flow.optimizer = optimizer  # so we can call flow.optimizer.step() later

# parent posterior (X1,X2); tracks P(X1 -> Y) and P(X2 -> Y)
ppost = ParentPosterior(d=2, sigma_eps=0.1, sigma_theta=1.0, prior_sparsity=0.3)

# MAIN LOOP
for t in range(30):  # 30 preference queries
    # 1) choose two random interventions (later: use PIG)
    x_choices, win_idx = dataset[t]  # assumes dataset already holds queries
    loser_idx = 1 - win_idx  # pairwise

    # 2) update PNF with this single comparison
    flow.train()
    
    if win_idx == 0:
        X_pair = torch.stack([x_choices[0], x_choices[1]])  # (2,d)
        label  = torch.tensor([1], dtype=torch.long)        # first wins
    else:
        X_pair = torch.stack([x_choices[1], x_choices[0]])  # put winner first
        label  = torch.tensor([1], dtype=torch.long)

    loss = -flow.logposterior((X_pair.T.unsqueeze(-1), label), weightprior=1.0)  # unsqueeze because prefflow expects (2,d,1) shape
    loss.backward(); flow.optimizer.step(); flow.optimizer.zero_grad()

    # 3) add observational point to parent posterior
    y_true = dataset.scm(x_choices[win_idx].unsqueeze(0))
    ppost.add_datapoint(x_choices[win_idx].unsqueeze(0), y_true)
    ppost.update_posterior()

    # 4) monitor every 5 iterations
    if (t+1) % 5 == 0:
        print(f"t={t+1}, loss={loss.item():.3f}, edge probs={ppost.edge_posterior().numpy()}")
        adj = np.array([[0,0,1],
                        [0,0,1],
                        [0,0,0]])
        edge_probs = ppost.edge_posterior().numpy()
        # build matrix for plot (pad zeros for Y row/col)
        prob_mat = np.array([[0,0,edge_probs[0]],
                            [0,0,edge_probs[1]],
                            [0,0,0]])
        plot_dag(adj, names=["X1","X2","Y"], edge_probs=prob_mat)