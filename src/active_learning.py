"""
active_learning.py
==================
Pool-based active learning for protein function classification.

The setup (a simulation of a low-label discovery setting):
- We actually have all labels, but we pretend most are hidden.
- We start from a small labeled "seed" set; the rest form an unlabeled pool.
- Each round, a *strategy* chooses which pool proteins to label next.
- We reveal those labels, retrain, and score on a fixed held-out test set.

Three strategies are provided, each with the SAME signature so the experiment
loop can treat them interchangeably:

    strategy(model, X_pool, n_select, rng) -> array of row indices into X_pool

- random_sampling:     pick n at random (the baseline to beat)
- uncertainty_sampling: pick the n the model is least confident about
- diversity_sampling:   pick uncertain AND spread-out proteins (cluster first)

Design notes:
- Model is LogisticRegression on ESM embeddings: fast to retrain many times,
  stable, and gives calibrated-enough probabilities for uncertainty.
- Strategies return *indices into the current pool*, not into the full dataset.
  The experiment loop (in the notebook) handles moving rows between sets.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_model(random_state: int = 42) -> "Pipeline":
    """LogReg on standardized features. Fast and stable for repeated retraining."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=random_state),
    )


# --------------------------------------------------------------------------- #
# Selection strategies  (all share the signature: model, X_pool, n_select, rng)
# --------------------------------------------------------------------------- #
def random_sampling(model, X_pool, n_select, rng):
    """Baseline: pick n_select pool rows uniformly at random. Ignores the model."""
    n_pool = X_pool.shape[0]
    n_select = min(n_select, n_pool)
    return rng.choice(n_pool, size=n_select, replace=False)


def uncertainty_sampling(model, X_pool, n_select, rng):
    """
    Pick the n_select proteins the model is least confident about.
    Uncertainty = 1 - max_class_probability  (high = unsure).
    """
    proba = model.predict_proba(X_pool)        # (n_pool, n_classes)
    uncertainty = 1.0 - proba.max(axis=1)      # (n_pool,)
    n_select = min(n_select, X_pool.shape[0])
    # argsort ascending, take the last n_select (the most uncertain).
    return np.argsort(uncertainty)[-n_select:]


def diversity_sampling(model, X_pool, n_select, rng, pool_factor: int = 5):
    """
    Uncertainty + diversity.

    Problem with pure uncertainty: the n most-uncertain proteins can be near-
    duplicates (homologous sequences), so we waste labels on redundant examples.

    Fix:
      1. Take a larger candidate set of the most-uncertain proteins
         (n_select * pool_factor of them).
      2. Cluster those candidates into n_select groups (KMeans on embeddings).
      3. From each cluster, pick its most-uncertain member.
    Result: n_select proteins that are both uncertain and spread across the
    sequence space.
    """
    n_pool = X_pool.shape[0]
    n_select = min(n_select, n_pool)

    proba = model.predict_proba(X_pool)
    uncertainty = 1.0 - proba.max(axis=1)

    # Step 1: candidate pool of most-uncertain proteins.
    n_candidates = min(n_select * pool_factor, n_pool)
    candidate_idx = np.argsort(uncertainty)[-n_candidates:]
    X_candidates = X_pool[candidate_idx]

    # If candidates barely exceed n_select, clustering adds nothing.
    if n_candidates <= n_select:
        return candidate_idx

    # Step 2: cluster candidates into n_select groups.
    seed = int(rng.integers(0, 1_000_000))
    km = KMeans(n_clusters=n_select, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(X_candidates)

    # Step 3: from each cluster, take its most-uncertain member.
    selected = []
    for c in range(n_select):
        members = np.where(cluster_labels == c)[0]
        if len(members) == 0:
            continue
        best_local = members[np.argmax(uncertainty[candidate_idx[members]])]
        selected.append(candidate_idx[best_local])

    return np.array(selected)


# Registry so the notebook can loop over strategies by name.
STRATEGIES = {
    "random": random_sampling,
    "uncertainty": uncertainty_sampling,
    "diversity": diversity_sampling,
}


# --------------------------------------------------------------------------- #
# One active learning run
# --------------------------------------------------------------------------- #
def run_active_learning(
    strategy_fn,
    X_labeled, y_labeled,        # initial seed set
    X_pool, y_pool,              # unlabeled pool (labels known but "hidden")
    X_test, y_test,              # fixed held-out test set
    n_rounds: int,
    n_per_round: int,
    random_state: int = 42,
):
    """
    Run one active learning experiment with a given strategy.

    Returns a list of dicts, one per round:
        {"n_labeled": int, "test_macro_f1": float}

    The loop is deliberately flat and readable:
      train -> evaluate -> select -> move rows -> repeat.
    """
    rng = np.random.default_rng(random_state)

    # Work on copies so the caller's arrays aren't mutated.
    X_lab, y_lab = X_labeled.copy(), y_labeled.copy()
    X_remaining, y_remaining = X_pool.copy(), y_pool.copy()

    history = []

    for round_num in range(n_rounds + 1):   # +1 to record the seed-only model
        # 1. Train on the currently labeled set.
        model = make_model(random_state)
        model.fit(X_lab, y_lab)

        # 2. Evaluate on the fixed test set.
        pred = model.predict(X_test)
        f1 = f1_score(y_test, pred, average="macro")
        history.append({"n_labeled": len(y_lab), "test_macro_f1": round(f1, 4)})

        # Stop once the pool is empty (or we've done all rounds).
        if round_num == n_rounds or len(y_remaining) == 0:
            break

        # 3. Strategy picks which pool rows to label next.
        pick = strategy_fn(model, X_remaining, n_per_round, rng)

        # 4. Move the picked rows from pool -> labeled set.
        X_lab = np.vstack([X_lab, X_remaining[pick]])
        y_lab = np.concatenate([y_lab, y_remaining[pick]])

        mask = np.ones(len(y_remaining), dtype=bool)
        mask[pick] = False
        X_remaining, y_remaining = X_remaining[mask], y_remaining[mask]

    return history
