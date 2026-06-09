"""
Replication of CV Inconsistency for DAG Structure Learning
Based on Corollary 6 of Lyu, Tai, Kolar, Aragam (AISTATS 2024)

We generate data from a known DAG, then compare:
- CV-selected lambda (should fail to recover true graph)
- BIC-selected lambda (should recover true graph)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from itertools import product

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. TRUE DAG DEFINITION
# ─────────────────────────────────────────────
# 5 nodes: X1 → X2, X1 → X3, X2 → X4, X3 → X4, X4 → X5
# True adjacency matrix B where B[i,j] != 0 means Xi → Xj
# Ordering is known: X1, X2, X3, X4, X5

TRUE_B = np.array([
    [0,   0.8, 0.6, 0,   0  ],   # X1 → X2, X1 → X3
    [0,   0,   0,   0.7, 0  ],   # X2 → X4
    [0,   0,   0,   0.5, 0  ],   # X3 → X4
    [0,   0,   0,   0,   0.9],   # X4 → X5
    [0,   0,   0,   0,   0  ],   # X5 has no children
])

TRUE_EDGES = set(zip(*np.where(TRUE_B != 0)))  # set of (i,j) pairs
p = 5  # number of nodes


def generate_dag_data(n, B, noise_std=1.0):
    """Generate n samples from a linear Gaussian DAG with coefficient matrix B."""
    p = B.shape[0]
    X = np.zeros((n, p))
    for j in range(p):
        parents = np.where(B[:, j] != 0)[0]
        X[:, j] = X[:, parents] @ B[parents, j] + np.random.normal(0, noise_std, n)
    return X


def compute_bic(X, j, parents, coefs, sigma2):
    n = X.shape[0]
    k = len(parents) + 1  # +1 for the noise variance parameter
    if k == 1:  # no parents, just noise
        residuals = X[:, j]
    else:
        residuals = X[:, j] - X[:, parents] @ coefs
    rss = np.sum(residuals ** 2)
    bic = n * np.log(rss / n) + k * np.log(n)
    return bic


def learn_dag_cv(X, ordering):
    """
    Learn DAG using Lasso with CV-selected lambda.
    For each node, regress onto its predecessors in the ordering.
    Returns estimated adjacency matrix.
    """
    n, p = X.shape
    B_hat = np.zeros((p, p))
    lambdas_chosen = []

    for j_idx in range(1, p):  # skip first node (no parents)
        j = ordering[j_idx]
        candidate_parents = ordering[:j_idx]

        X_parents = X[:, candidate_parents]
        y = X[:, j]

        # CV to select lambda
        lasso_cv = LassoCV(cv=5, max_iter=10000, fit_intercept=False)
        lasso_cv.fit(X_parents, y)
        coefs = lasso_cv.coef_
        lambdas_chosen.append(lasso_cv.alpha_)

        for k, parent in enumerate(candidate_parents):
            if abs(coefs[k]) > 1e-6:
                B_hat[parent, j] = coefs[k]

    return B_hat, lambdas_chosen


def learn_dag_bic(X, ordering, n_lambdas=50):
    """
    Learn DAG using Lasso with BIC-selected lambda.
    For each node, regress onto its predecessors in the ordering.
    Returns estimated adjacency matrix.
    """
    n, p = X.shape
    B_hat = np.zeros((p, p))
    lambdas_chosen = []

    for j_idx in range(1, p):
        j = ordering[j_idx]
        candidate_parents = ordering[:j_idx]

        X_parents = X[:, candidate_parents]
        y = X[:, j]

        # Grid of lambdas
        lambda_max = np.max(np.abs(X_parents.T @ y)) / n
        lambdas = np.logspace(np.log10(lambda_max * 1e-3), np.log10(lambda_max), n_lambdas)[::-1]

        best_bic = np.inf
        best_coefs = np.zeros(len(candidate_parents))
        best_lambda = lambdas[0]

        for lam in lambdas:
            lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
            lasso.fit(X_parents, y)
            coefs = lasso.coef_
            parents_idx = np.where(np.abs(coefs) > 1e-6)[0]
            sigma2 = np.var(y - X_parents @ coefs)
            bic = compute_bic(X, j, parents_idx, coefs[parents_idx], sigma2)
            if bic < best_bic:
                best_bic = bic
                best_coefs = coefs.copy()
                best_lambda = lam

        lambdas_chosen.append(best_lambda)
        for k, parent in enumerate(candidate_parents):
            if abs(best_coefs[k]) > 1e-6:
                B_hat[parent, j] = best_coefs[k]

    return B_hat, lambdas_chosen


def compute_shd(B_true, B_hat):
    """Structural Hamming Distance: number of edge insertions/deletions needed."""
    true_edges = set(zip(*np.where(B_true != 0))) if np.any(B_true != 0) else set()
    hat_edges = set(zip(*np.where(B_hat != 0))) if np.any(B_hat != 0) else set()
    # Edges in hat but not true (false positives) + edges in true but not hat (false negatives)
    return len(hat_edges - true_edges) + len(true_edges - hat_edges)


def compute_fdr(B_true, B_hat):
    """False Discovery Rate: fraction of estimated edges that are wrong."""
    true_edges = set(zip(*np.where(B_true != 0))) if np.any(B_true != 0) else set()
    hat_edges = set(zip(*np.where(B_hat != 0))) if np.any(B_hat != 0) else set()
    if len(hat_edges) == 0:
        return 0.0
    return len(hat_edges - true_edges) / len(hat_edges)


def compute_tpr(B_true, B_hat):
    """True Positive Rate: fraction of true edges recovered."""
    true_edges = set(zip(*np.where(B_true != 0))) if np.any(B_true != 0) else set()
    hat_edges = set(zip(*np.where(B_hat != 0))) if np.any(B_hat != 0) else set()
    if len(true_edges) == 0:
        return 1.0
    return len(true_edges & hat_edges) / len(true_edges)


# ─────────────────────────────────────────────
# 2. SIMULATION ACROSS SAMPLE SIZES
# ─────────────────────────────────────────────
ordering = list(range(p))  # known ordering: 0,1,2,3,4
sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
n_trials = 20  # trials per sample size

results = {
    'cv':  {'shd': [], 'fdr': [], 'tpr': []},
    'bic': {'shd': [], 'fdr': [], 'tpr': []},
}

print("Running simulation...")
for n in sample_sizes:
    shd_cv, fdr_cv, tpr_cv = [], [], []
    shd_bic, fdr_bic, tpr_bic = [], [], []

    for trial in range(n_trials):
        X = generate_dag_data(n, TRUE_B)

        B_cv, _ = learn_dag_cv(X, ordering)
        B_bic, _ = learn_dag_bic(X, ordering)

        shd_cv.append(compute_shd(TRUE_B, B_cv))
        fdr_cv.append(compute_fdr(TRUE_B, B_cv))
        tpr_cv.append(compute_tpr(TRUE_B, B_cv))

        shd_bic.append(compute_shd(TRUE_B, B_bic))
        fdr_bic.append(compute_fdr(TRUE_B, B_bic))
        tpr_bic.append(compute_tpr(TRUE_B, B_bic))

    results['cv']['shd'].append(np.mean(shd_cv))
    results['cv']['fdr'].append(np.mean(fdr_cv))
    results['cv']['tpr'].append(np.mean(tpr_cv))

    results['bic']['shd'].append(np.mean(shd_bic))
    results['bic']['fdr'].append(np.mean(fdr_bic))
    results['bic']['tpr'].append(np.mean(tpr_bic))

    print(f"n={n:5d} | CV SHD={np.mean(shd_cv):.2f}, FDR={np.mean(fdr_cv):.2f}, TPR={np.mean(tpr_cv):.2f} "
          f"| BIC SHD={np.mean(shd_bic):.2f}, FDR={np.mean(fdr_bic):.2f}, TPR={np.mean(tpr_bic):.2f}")


# ─────────────────────────────────────────────
# 3. PLOT RESULTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("CV vs BIC for DAG Structure Learning (p=5, known ordering)", fontsize=13)

metrics = ['shd', 'fdr', 'tpr']
labels = ['SHD (lower is better)', 'FDR (lower is better)', 'TPR (higher is better)']

for ax, metric, label in zip(axes, metrics, labels):
    ax.plot(sample_sizes, results['cv'][metric],  'o-', color='steelblue', label='CV')
    ax.plot(sample_sizes, results['bic'][metric], 's--', color='darkorange', label='BIC')
    ax.set_xscale('log')
    ax.set_xlabel('Sample size n (log scale)')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # For SHD and FDR, draw a zero line showing perfect recovery
    if metric in ['shd', 'fdr']:
        ax.axhline(0, color='black', linestyle='-.', linewidth=1, label='Perfect')

plt.tight_layout()
plt.savefig('dag_cv_inconsistency.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to dag_cv_inconsistency.png")