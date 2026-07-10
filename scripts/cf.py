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
    """Load ratings.csv into parallel numpy arrays.

    Returns:
        (users, items, ratings, times) — four aligned arrays, one entry per
        rating row: raw MovieLens userId, raw movieId, the 0.5–5.0 star
        rating, and the unix timestamp. Ids are NOT remapped here; use
        build_matrix() for contiguous indices.
    """
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
    """Load movies.csv genre labels.

    Returns:
        Mapping of raw movieId -> list of genre strings (MovieLens's
        pipe-separated field split apart; "(no genres listed)" comes
        through as a literal single-element list).
    """
    with open(ML / "movies.csv", newline="") as f:
        return {int(r["movieId"]): r["genres"].split("|") for r in csv.DictReader(f)}


def temporal_split(users, items, ratings, times, test_frac=0.2):
    """Split ratings chronologically PER USER (no global time cutoff).

    Each user's ratings are sorted by timestamp; the earliest
    (1 - test_frac) go to train and the latest test_frac to test. This
    mimics the real deployment question — "given what a user rated so far,
    predict what they rate next" — and prevents future->past leakage
    within a user.

    Args:
        users/items/ratings/times: aligned arrays from load_ratings().
        test_frac: fraction of each user's LATEST ratings held out.

    Returns:
        Boolean mask over the rating rows — True = train, False = test.
        Every user keeps at least 1 training rating.
    """
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
    """Build the sparse user x item rating matrix with contiguous indices.

    Raw MovieLens ids are sparse (movieIds go to ~193k for 9.7k movies), so
    rows/columns are remapped to dense 0..n-1 indices.

    Args:
        users/items/ratings: aligned arrays (a train subset is fine).

    Returns:
        (X, uids, iids): X is a scipy CSR matrix [n_users, n_items] of raw
        star ratings; uids/iids are sorted arrays mapping matrix index ->
        original MovieLens id (so iids[j] recovers the movieId of column j).
    """
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
    """Train item-item similarities: adjusted cosine + support shrinkage.

    Each rating is centered by its USER's mean (adjusted cosine, the best
    variant in Sarwar et al. 2001), then item columns are compared by
    cosine. Raw cosines are shrunk toward 0 by n/(n+beta), n = number of
    co-raters — with 610 users, cosine over a handful of co-raters is
    noise, and the eval shows beta wants to be large (~400; EVALUATION.md
    finding 3). Per item, the top_k strongest-|sim| neighbors are kept.

    Args:
        X: CSR user x item rating matrix (from build_matrix()).
        beta: shrinkage strength; 0 disables shrinkage.
        top_k: neighbors kept per item.

    Returns:
        (nbr_idx, nbr_sim): int32 [n_items, top_k] neighbor column indices
        (-1 padding where an item has fewer than top_k neighbors) and
        float32 shrunk similarities (0.0 padding), sorted by |sim| desc.
    """
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
    """Score ALL items for one user by folding their ratings into the model.

    Mirrors the runtime exactly: for each candidate j, walk j's OWN
    truncated top-K neighbor list and accumulate over the neighbors the
    user has rated:

        acc_j = sum sim(i,j) * (r_ui - user_mean)     (i rated by user)
        den_j = sum |sim(i,j)|

    normalize=True returns acc/den — a rating PREDICTION, which the eval
    shows is catastrophic for top-N ranking (HR@10 0.42 -> 0.01,
    EVALUATION.md finding 2). normalize=False returns the raw acc sum,
    which is what ships. Kept switchable so the eval can report both.

    Args:
        nbr_idx, nbr_sim: neighbor arrays from train_item_item().
        user_ratings: {item_index: star_rating} for this user's rated set.
        user_mean: the user's mean rating (damped at runtime for +-1 data).
        n_items: total item count (score vector length).
        normalize: divide by den (True) or return the raw sum (False).

    Returns:
        Float array [n_items]; rated items and candidates with zero rated
        neighbors are -inf (unrankable), everything else is the fold-in
        score, higher = more recommended.
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
    """Compute hit-rate@10 and NDCG@10 for one user's ranked list.

    Args:
        top10: item indices of the user's top-10 recommendations, best first.
        relevant: item indices of the user's held-out relevant items
            (test ratings >= REL_THRESHOLD).

    Returns:
        (hr, ndcg): hr is 1.0 if ANY relevant item appears in the top-10,
        else 0.0. ndcg uses binary relevance with the standard log2
        discount, normalized by the ideal DCG for min(|relevant|, 10)
        items — so a user with one relevant item can still score 1.0 by
        ranking it first.
    """
    hits = [1.0 if j in relevant else 0.0 for j in top10]
    hr = 1.0 if any(hits) else 0.0
    dcg = sum(h / np.log2(k + 2) for k, h in enumerate(hits))
    idcg = sum(1.0 / np.log2(k + 2) for k in range(min(len(relevant), 10)))
    return hr, (dcg / idcg if idcg > 0 else 0.0)
