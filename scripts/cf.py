"""Item-item collaborative filtering core (Sarwar et al. 2001 family).

Training: adjusted cosine — center each rating by its USER's mean, then
cosine between item columns (full-norm variant, i.e. exactly the cosine of
centered item vectors). Similarities are shrunk toward 0 by n/(n+beta)
where n = co-rater count, damping low-support pairs. Top-K neighbors kept.

Evaluation: temporal split (last test_frac of each user's ratings held out),
fold-in scoring exactly the way the runtime does it — over the CANDIDATE's
own truncated neighbor list intersected with the user's rated set.
"""

import csv
from pathlib import Path

import numpy as np
from scipy import sparse

ML = Path(__file__).resolve().parent.parent / "data" / "movielens"

TOP_K = 50
REL_THRESHOLD = 4.0  # held-out rating >= this counts as "relevant"


# ---------- data ----------

def load_ratings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ratings.csv -> (user_idx, item_idx, rating, timestamp) plus id maps."""
    users, items, ratings, times = [], [], [], []
    with open(ML / "ratings.csv", newline="") as f:
        for row in csv.DictReader(f):
            users.append(int(row["userId"]))
            items.append(int(row["movieId"]))
            ratings.append(float(row["rating"]))
            times.append(int(row["timestamp"]))
    return (np.array(users), np.array(items), np.array(ratings, dtype=np.float64),
            np.array(times, dtype=np.int64))


def load_genres() -> dict[int, list[str]]:
    with open(ML / "movies.csv", newline="") as f:
        return {int(r["movieId"]): r["genres"].split("|") for r in csv.DictReader(f)}


def temporal_split(users, items, ratings, times, test_frac=0.2):
    """Per user: earliest (1-test_frac) -> train, latest test_frac -> test."""
    train_mask = np.zeros(len(users), dtype=bool)
    order = np.lexsort((times, users))  # by user, then time
    start = 0
    for end in np.flatnonzero(np.diff(users[order])) + 1:
        idx = order[start:end]
        n_train = max(1, int(round(len(idx) * (1 - test_frac))))
        train_mask[idx[:n_train]] = True
        start = end
    idx = order[start:]
    train_mask[idx[: max(1, int(round(len(idx) * (1 - test_frac))))]] = True
    return train_mask


# ---------- training ----------

def build_matrix(users, items, ratings):
    """Sparse user x item matrix with contiguous index maps."""
    uids = np.unique(users)
    iids = np.unique(items)
    umap = {u: k for k, u in enumerate(uids)}
    imap = {i: k for k, i in enumerate(iids)}
    X = sparse.csr_matrix(
        (ratings, ([umap[u] for u in users], [imap[i] for i in items])),
        shape=(len(uids), len(iids)),
    )
    return X, uids, iids


def train_item_item(X: sparse.csr_matrix, beta: float, top_k: int = TOP_K):
    """-> (idx, sim): [n_items, top_k] neighbor indices and shrunk sims."""
    # center each user's ratings by their mean (adjusted cosine)
    counts = np.diff(X.indptr)
    means = np.divide(X.sum(axis=1).A1, counts,
                      out=np.zeros(X.shape[0]), where=counts > 0)
    Xc = X.copy().astype(np.float64)
    Xc.data -= np.repeat(means, counts)

    S = (Xc.T @ Xc).tocsr()                      # centered dot products
    norms = np.sqrt(S.diagonal())
    B = X.copy()
    B.data = np.ones_like(B.data)
    N = (B.T @ B).tocsr()                        # co-rater counts

    n_items = X.shape[1]
    nbr_idx = np.full((n_items, top_k), -1, dtype=np.int32)
    nbr_sim = np.zeros((n_items, top_k), dtype=np.float32)

    for i in range(n_items):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        cols, vals = S.indices[lo:hi], S.data[lo:hi]
        keep = (cols != i) & (norms[cols] > 0)
        cols, vals = cols[keep], vals[keep]
        if norms[i] == 0 or len(cols) == 0:
            continue
        co = np.asarray(N[i, cols].todense()).ravel()
        sims = vals / (norms[i] * norms[cols]) * (co / (co + beta) if beta > 0 else 1.0)
        top = np.argsort(-np.abs(sims))[:top_k]  # strongest |sim| first
        nbr_idx[i, : len(top)] = cols[top]
        nbr_sim[i, : len(top)] = sims[top]
    return nbr_idx, nbr_sim


# ---------- fold-in scoring (mirrors the runtime) ----------

def fold_in_scores(nbr_idx, nbr_sim, user_ratings, user_mean, n_items, normalize=True):
    """Score ALL items for one user from their rated set.

    For each candidate j: over (i, s) in j's own neighbor list with i rated:
      acc_j = sum s * (r_ui - mean_u),  den_j = sum |s|
    normalize=True -> acc/den (spec formula), else raw acc.
    Rated items are excluded (set to -inf).
    """
    dev = np.zeros(n_items)
    rated = np.zeros(n_items, dtype=bool)
    for i, r in user_ratings.items():
        dev[i] = r - user_mean
        rated[i] = True

    valid = nbr_idx >= 0
    safe_idx = np.where(valid, nbr_idx, 0)
    hit = valid & rated[safe_idx]                       # [n_items, K]
    acc = np.where(hit, nbr_sim * dev[safe_idx], 0.0).sum(axis=1)
    if normalize:
        den = np.where(hit, np.abs(nbr_sim), 0.0).sum(axis=1)
        scores = np.divide(acc, den, out=np.zeros_like(acc), where=den > 0)
        scores[den == 0] = -np.inf                      # CF-silent: unrankable
    else:
        scores = acc
        scores[~hit.any(axis=1)] = -np.inf
    scores[rated] = -np.inf
    return scores


# ---------- metrics ----------

def hr_ndcg_at10(top10: np.ndarray, relevant: set[int]) -> tuple[float, float]:
    hits = [1.0 if j in relevant else 0.0 for j in top10]
    hr = 1.0 if any(hits) else 0.0
    dcg = sum(h / np.log2(k + 2) for k, h in enumerate(hits))
    idcg = sum(1.0 / np.log2(k + 2) for k in range(min(len(relevant), 10)))
    return hr, (dcg / idcg if idcg > 0 else 0.0)
