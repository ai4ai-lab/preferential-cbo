import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from sklearn.preprocessing import StandardScaler

class CausalDiscoveryBaselines:
    """Collection of baseline causal discovery methods"""
    def __init__(self, n_nodes, device='cpu'):
        self.n_nodes = n_nodes
        self.device = device
        self.data_buffer = []

    # -------- Data Management --------
    def add_data(self, intervention_node, outcome):
        out_np = outcome.detach().cpu().numpy() if torch.is_tensor(outcome) else np.asarray(outcome)
        self.data_buffer.append({'intervention_node': int(intervention_node), 'outcome': out_np})
    
    def get_data_matrix(self):
        if not self.data_buffer:
            return None, None
        X = np.array([d['outcome'] for d in self.data_buffer])
        interventions = np.array([d['intervention_node'] for d in self.data_buffer])
        if X.ndim == 1:
            X = X.reshape(-1, self.n_nodes)
        return X, interventions

    # -------- Baseline Methods --------
    def random_baseline(self, sparsity=0.3, seed=42):
        """Random DAG with given sparsity"""
        rng = np.random.default_rng(seed)
        adj = (rng.random((self.n_nodes, self.n_nodes)) < float(sparsity)).astype(int)
        adj = np.triu(adj, k=1)  # Ensure upper triangular (DAG)
        return adj
    
    def fully_connected_baseline(self):
        """Fully connected DAG (upper triangular)"""
        adj = np.ones((self.n_nodes, self.n_nodes))
        adj = np.triu(adj, k=1)
        return adj.astype(int)
    
    def lasso_baseline(self, alpha=0.1):
        """
        LASSO regression for each node to identify parents.
        Simple but often effective baseline.
        """
        X, interventions = self.get_data_matrix()
        if X is None or len(X) < 10:
            return self.random_baseline()
        
        adj = np.zeros((self.n_nodes, self.n_nodes))
        
        for target in range(self.n_nodes):
            # Only use data where target wasn't intervened on
            if interventions is not None:
                mask_valid = (interventions != target)
                X_valid = X[mask_valid]
                if len(X_valid) < 5:
                    continue
            else:
                X_valid = X
                
            mask_features = np.ones(self.n_nodes, dtype=bool)
            mask_features[target] = False
            
            X_parents = X_valid[:, mask_features]
            y_target = X_valid[:, target]

            # Standardize features
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_parents)
            y_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1)).ravel()
            
            # LASSO with cross-validation
            try:
                lasso = LassoCV(cv=5, max_iter=1000, random_state=42)
                lasso.fit(X_scaled, y_scaled)
                
                # Non-zero coefficients indicate edges
                parent_idx = np.where(mask_features)[0]
                for i, coef in enumerate(lasso.coef_):
                    if abs(coef) > 0.01:  # Threshold for considering an edge
                        adj[parent_idx[i], target] = 1
            except:
                # If LASSO fails, use correlation
                for i, parent in enumerate(parent_idx):
                    corr = np.corrcoef(X[:, parent], y_target)[0, 1]
                    if abs(corr) > 0.3:
                        adj[parent, target] = 1
                        
        return adj.astype(int)
    
    def pc_skeleton(self, alpha=0.05):
        """
        Simplified PC algorithm: finds skeleton using correlation tests.
        Note: This is a simplified version without orientation rules.
        """
        X, interventions = self.get_data_matrix()
        if X is None or len(X) < 20:
            return self.random_baseline()
        
        adj = np.ones((self.n_nodes, self.n_nodes)) - np.eye(self.n_nodes)
        
        # Test marginal independence
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if interventions is not None:
                    mask = (interventions != i) & (interventions != j)
                    X_obs = X[mask]
                    if len(X_obs) < 10:
                        continue
                else:
                    X_obs = X
                    
                corr, p_value = stats.pearsonr(X_obs[:, i], X_obs[:, j])
                if p_value > alpha:
                    adj[i, j] = 0
                    adj[j, i] = 0
        
        # Test conditional independence (order 1)
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if adj[i, j] == 0:
                    continue

                if interventions is not None:
                    mask_ij = (interventions != i) & (interventions != j)
                    X_for_test = X[mask_ij]
                    if len(X_for_test) < 10:
                        continue
                else:
                    X_for_test = X
                    
                # Test conditioning on each other variable
                for k in range(self.n_nodes):
                    if k == i or k == j:
                        continue

                    # Exclude interventions on k for conditioning
                    if interventions is not None:
                        mask_final = mask_ij & (interventions != k)
                        X_final = X[mask_final]
                        if len(X_final) < 10:
                            continue
                    else:
                        X_final = X_for_test
                    
                    try:
                        # Calculate partial correlation
                        resid_i = X_final[:, i] - LinearRegression().fit(
                            X_final[:, k].reshape(-1, 1), X_final[:, i]
                        ).predict(X_final[:, k].reshape(-1, 1))
                        
                        resid_j = X_final[:, j] - LinearRegression().fit(
                            X_final[:, k].reshape(-1, 1), X_final[:, j]
                        ).predict(X_final[:, k].reshape(-1, 1))
                        
                        corr, p_value = stats.pearsonr(resid_i, resid_j)
                        if p_value > alpha:
                            adj[i, j] = 0
                            adj[j, i] = 0
                            break
                    except:
                        pass
        
        return self._orient_skeleton(adj)
    
    def _orient_skeleton(self, skeleton):
        """Orient edges in skeleton to form DAG"""
        X, interventions = self.get_data_matrix()
        if X is None:
            return np.triu(skeleton, k=1)
        
        # Calculate variance only from non-intervened data for each node
        variances = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            if interventions is not None:
                mask = (interventions != i)
                if mask.sum() > 0:
                    variances[i] = np.var(X[mask, i])
                else:
                    variances[i] = np.var(X[:, i])  # fallback
            else:
                variances[i] = np.var(X[:, i])
        
        order = np.argsort(-variances)
        
        # Create DAG following this order
        adj = np.zeros_like(skeleton)
        for i, node_i in enumerate(order):
            for j, node_j in enumerate(order[i+1:], i+1):
                if skeleton[node_i, node_j] > 0 or skeleton[node_j, node_i] > 0:
                    adj[node_i, node_j] = 1
                    
        return adj.astype(int)
    
    def notears_linear(self, max_iter=200, thresh=0.10, k_max=3, random_state=42):
        """
        NOTEARS-lite (surrogate): per-target ElasticNet with intervene-masking,
        z-scoring, hard coef threshold, greedy acyclic projection with indegree cap.
        """
        X, interventions = self.get_data_matrix()
        if X is None or len(X) < 20:
            return self.random_baseline()

        n, d = X.shape
        W = np.zeros((d, d), dtype=float)

        for j in range(d):
            mask = (interventions != j) if interventions is not None else np.ones(n, dtype=bool)
            Xj = X[mask]
            if Xj.shape[0] < 10:
                continue
            y = Xj[:, j]
            Z = Xj[:, np.arange(d) != j]

            Zm = Z.mean(axis=0, keepdims=True)
            Zs = Z.std(axis=0, keepdims=True) + 1e-8
            Zz = (Z - Zm) / Zs
            y_c = y - y.mean()

            try:
                en = ElasticNetCV(l1_ratio=[0.8, 0.9, 1.0],  # includes LASSO
                                cv=5, max_iter=max_iter, random_state=random_state)
                en.fit(Zz, y_c)
                coef = en.coef_
            except Exception:
                continue

            idx_other = [k for k in range(d) if k != j]
            W[idx_other, j] = coef

        W[np.abs(W) < thresh] = 0.0

        # Greedy DAG projection with in-degree cap
        A = np.zeros((d, d), dtype=int)
        indeg = np.zeros(d, dtype=int)
        edges = [(abs(W[i, j]), i, j) for i in range(d) for j in range(d) if i != j and W[i, j] != 0.0]
        edges.sort(reverse=True)

        def _is_cyclic(A):
            A = (A > 0).astype(int)
            vis = [0]*d
            adj = {i: set(np.where(A[i, :])[0]) for i in range(d)}
            def dfs(u):
                vis[u] = 1
                for v in adj[u]:
                    if vis[v] == 1: return True
                    if vis[v] == 0 and dfs(v): return True
                vis[u] = 2
                return False
            return any(dfs(i) for i in range(d) if vis[i] == 0)

        for w, i, j in edges:
            if indeg[j] >= k_max: 
                continue
            A[i, j] = 1
            if _is_cyclic(A):
                A[i, j] = 0
            else:
                indeg[j] += 1

        return A
    
    def causal_sufficiency(self, alpha=0.05, min_per_group=10, min_abs_shift=0.15,
                           use_fdr=True, only_direct=True):
        """
        Heuristic: add i->j if do(X_i) shifts X_j (Welch t-test) with clean controls,
        effect-size filter, and light transitive pruning.
        """
        X, interventions = self.get_data_matrix()
        if X is None or len(X) < 30:
            return self.random_baseline()

        d = self.n_nodes
        potential = np.zeros((d, d), dtype=float)  # store a 'strength' score

        # First pass: per (i -> j) test with clean controls
        pvals = []
        pairs = []
        shifts = []  # mean shift |E[X_j | do(i)] - E[X_j | not i, not j]|

        for i in range(d):
            treated = (interventions == i)
            if treated.sum() < min_per_group:
                continue
            X_t = X[treated]

            for j in range(d):
                if j == i:
                    continue
                control = (interventions != i) & (interventions != j)
                if control.sum() < min_per_group:
                    continue
                X_c = X[control]

                # test on X_j
                y_t = X_t[:, j]
                y_c = X_c[:, j]

                # Welch t-test
                tstat, p = stats.ttest_ind(y_t, y_c, equal_var=False)
                # effect size: absolute mean shift (units of variable scale)
                shift = float(np.abs(y_t.mean() - y_c.mean()))

                pvals.append(p)
                pairs.append((i, j))
                shifts.append(shift)

        # Multiple-testing control
        keep = np.ones(len(pvals), dtype=bool)
        if len(pvals) and use_fdr:
            # Benjamini–Hochberg
            order = np.argsort(pvals)
            m = len(pvals)
            thresh = np.array([(k+1)/m * alpha for k in range(m)], dtype=float)
            passed = np.zeros(m, dtype=bool)
            running = 0
            for rank, idx in enumerate(order):
                if pvals[idx] <= thresh[rank]:
                    running = rank
                    passed[rank] = True
            # allow all up to last passing rank
            if passed.any():
                last = np.max(np.nonzero(passed))
                mask_bh = np.zeros(m, dtype=bool); mask_bh[order[:last+1]] = True
                keep &= mask_bh
            else:
                keep &= False
        else:
            keep &= (np.array(pvals) <= alpha)

        # effect size filter
        keep &= (np.array(shifts) >= min_abs_shift)

        for k, flag in enumerate(keep):
            if not flag: 
                continue
            i, j = pairs[k]
            # use a simple 'strength' = shift / (p + eps)
            potential[i, j] = shifts[k] / (pvals[k] + 1e-12)

        if not only_direct:
            return (potential > 0).astype(int)

        # Second pass: evidence-based pruning of indirects
        adj = (potential > 0).astype(int)
        # Sort candidate 2-hop paths roughly by weakest link strength
        for i in range(d):
            for j in range(d):
                if i == j or adj[i, j] == 0:
                    continue
                # look for mediators k
                for k in range(d):
                    if k == i or k == j: 
                        continue
                    if adj[i, k] and adj[k, j]:
                        # Only drop i->j if its 'strength' is clearly weaker than the 2-hop path
                        if potential[i, j] < min(potential[i, k], potential[k, j]):
                            # Conditional re-test: is i->j still significant given k?
                            # Quick regression residual check:
                            try:
                                y = X[interventions != j, j]
                                Xi = X[interventions != j, i]
                                Xk = X[interventions != j, k].reshape(-1, 1)
                                # regress out k
                                resid_y = y - LinearRegression().fit(Xk, y).predict(Xk)
                                resid_i = Xi - LinearRegression().fit(Xk, Xi).predict(Xk)
                                _, p_cond = stats.pearsonr(resid_i, resid_y)
                                if p_cond > alpha:
                                    adj[i, j] = 0
                                    break
                            except Exception:
                                # Fall back to strength comparison only
                                adj[i, j] = 0
                                break

        return adj.astype(int)
    
    def _project_to_dag(self, W):
        """Project weight matrix to DAG by removing cycles"""
        # Simple heuristic: threshold and order by weight magnitude
        threshold = np.percentile(np.abs(W[W != 0]), 30) if np.any(W != 0) else 0.1
        W_thresh = W * (np.abs(W) > threshold)
        
        # Remove cycles by keeping only upper triangular after reordering
        # Order nodes by total absolute incoming weight
        scores = np.sum(np.abs(W_thresh), axis=0)
        order = np.argsort(scores)
        
        W_ordered = W_thresh[order][:, order]
        W_dag = np.triu(W_ordered, k=1)
        
        # Reorder back
        W_final = np.zeros_like(W)
        for i, oi in enumerate(order):
            for j, oj in enumerate(order):
                W_final[oi, oj] = W_dag[i, j]
                
        return W_final