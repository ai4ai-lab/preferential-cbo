# Preferential Causal Bayesian Optimisation (PCBO)

PCBO is a framework for **causal discovery from preferences**. Instead of relying on numeric outcomes of interventions, PCBO learns from **comparative judgements** (e.g. “A is better than B”), combining:

- **Bayesian local structure learning** (parent posteriors per node),
- **Preferential Normalising Flows (PNFs)** for flexible utility modelling from comparisons,
- **Acquisition** that balances structure discovery and preference learning.

PCBO targets settings where calibrated outcomes are scarce or noisy, but comparisons are feasible (medicine, policy, user studies). A full description and empirical study appear in the accompanying MSc thesis.



## Main Features

- **Local Bayesian structure learning**  
  Exact conjugate updates for small graphs and a **scalable MCMC variant** for larger ones, returning edge marginals $P(j\ \to\ i)$. A greedy DAG projection produces acyclic graphs for inspection and metrics.

- **Preferential flows for utilities**  
  A **PNF** equates latent utility with the flow log-density and links pairwise outcomes via a **logistic likelihood**. A learnable temperature \(s\) controls sharpness.

- **Acquisition**  
  Pairwise **PIG-style** uncertainty from the flow and **EEIG** from expected edge-entropy reduction; blended across PCBO rounds.

- **End-to-end loop**  
  Update flow → refresh local posteriors → select intervention → record preference → repeat. Evaluation uses edge metrics and graph visualisation.



## Repository Structure

```bash
preferential-cbo/
├── demo_pcbo.ipynb   # Main notebook: run PCBO end-to-end
├── datasets.py   # Toy, ER, and medical-style generators
├── acquisition.py   # PIG / EEIG scoring utilities
├── prefflow.py   # Preferential flow wrapper and training objective
├── flows.py   # RealNVP / Residual / Neural Spline builders
├── likelihood.py   # Pairwise and k-way preference likelihoods
├── parent_posterior.py   # Exact local posterior + greedy DAG projection
├── parent_posterior_scalable.py   # MCMC-based scalable local posterior
├── baselines.py   # Random, LASSO, PC-lite, NOTEARS-lite, interventional t-tests
├── figures/   # Generated plots
└── requirements.txt
```



## Installation

```bash
git clone https://github.com/ai4ai-lab/preferential-cbo.git
cd preferential-cbo
pip install -r requirements.txt   # Python 3.9+ recommended
```

Dependencies: PyTorch, normflows, NumPy, SciPy, Matplotlib.



## Quick Start

Run the demo notebook:

```bash
jupyter notebook demo_pcbo.ipynb
```

The notebook covers:
1. Choose a dataset (3-node, 6-node, Erdős–Rényi, medical).
2. Run PCBO for a set number of rounds.
3. Inspect results: edge-probability matrix, projected DAG, preference accuracy, plots.



## Datasets

- **Toy DAGs** (3- and 6-node): simple, interpretable, fast smoke tests.
- **Erdős–Rényi DAGs**: scalable linear-Gaussian graphs for stress tests.
- **Medical-style case**: interpretable lifestyle → biomarkers → outcome DAG.

All datasets simulate interventions $\mathrm{do}(X_i=v)$ and provide ground truth for evaluation.

## Method Overview

- **Locals**: Bayesian updates per node; exact for small graphs, MCMC for larger.
- **Flows**: utilities from log-density; preferences follow logistic likelihood.
- **Acquisition**: blend of preference uncertainty and edge entropy reduction.
- **Thresholding**: F1-based (with labels), Beta-mixture (label-free), or manual.
- **Visualisation**: graphs with edge width proportional to confidence.

## Baselines
- **Random / Fully connected**
- **LASSO regression**
- **PC-lite skeleton**
- **NOTEARS-lite surrogate**
- **Interventional t-tests**

These provide reference points; PCBO’s strength is to achieve comparable or better discovery while relying only on preference data.

## Reproducibility
1. Install dependencies.
2. Run ```demo_pcbo.ipynb``` with your dataset of choice.
3. Export figures and metrics using built-in plotting functions.
4. For ablations, swap flow backbone (RealNVP / Residual / Neural Spline).

## Acknowledgements
Developed at the AI for Actionable Impact (AI4AI) Lab, Imperial College London.
Thanks to Prof. Sonali Parbhoo for supervision and Marcello Negri for contributions and feedback.
