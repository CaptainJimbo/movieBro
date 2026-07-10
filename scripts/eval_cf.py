#!/usr/bin/env python3
"""Step-2 eval: item-item CF vs popularity baselines on a temporal split.

Per user: earliest 80% of ratings -> train, latest 20% -> test.
Relevant = held-out rating >= 4.0. Metrics: hit-rate@10, NDCG@10, averaged
over users with >= 1 relevant test item. Prints a markdown table for
EVALUATION.md.

Rows: popularity, genre-matched popularity, CF x shrinkage beta grid x
{normalized, unnormalized} fold-in.
"""

import sys
from collections import Counter, defaultdict

import numpy as np

from cf import (REL_THRESHOLD, build_matrix, fold_in_scores, hr_ndcg_at10,
                load_genres, load_ratings, temporal_split)

BETAS = [0, 5, 25, 100]


def main() -> int:
    """Run the full step-2 evaluation and print a markdown results table.

    Pipeline: load ratings -> per-user temporal split -> build the train
    matrix -> evaluate the two popularity baselines, then item-item CF
    across the BETAS shrinkage grid x {normalized, unnormalized} fold-in.
    Per-method progress goes to stderr; the final markdown table (for
    EVALUATION.md) goes to stdout.

    Returns:
        Process exit code (0 on success).
    """
    users, items, ratings, times = load_ratings()
    train_mask = temporal_split(users, items, ratings, times)

    X, uids, iids = build_matrix(users[train_mask], items[train_mask], ratings[train_mask])
    n_users, n_items = X.shape
    umap = {u: k for k, u in enumerate(uids)}
    imap = {i: k for k, i in enumerate(iids)}
    print(f"train: {train_mask.sum()} ratings | test: {(~train_mask).sum()} "
          f"| {n_users} users x {n_items} items", file=sys.stderr)

    # per-user train ratings/means and test relevant sets (test items must be
    # mappable, i.e. seen in train by SOMEONE — else no method can rank them)
    train_by_user: dict[int, dict[int, float]] = defaultdict(dict)
    for u, i, r in zip(users[train_mask], items[train_mask], ratings[train_mask]):
        train_by_user[umap[u]][imap[i]] = r
    relevant_by_user: dict[int, set[int]] = defaultdict(set)
    n_unmappable = 0
    for u, i, r in zip(users[~train_mask], items[~train_mask], ratings[~train_mask]):
        if r >= REL_THRESHOLD:
            if i in imap:
                relevant_by_user[umap[u]].add(imap[i])
            else:
                n_unmappable += 1
    eval_users = [u for u in range(n_users) if relevant_by_user[u]]
    print(f"eval users: {len(eval_users)} (with >=1 relevant held-out item); "
          f"{n_unmappable} relevant test ratings on items absent from train "
          f"(unreachable by any method)", file=sys.stderr)

    # ---- baselines ----
    pop = np.zeros(n_items)
    item_counts = Counter(imap[i] for i in items[train_mask])
    for j, c in item_counts.items():
        pop[j] = c

    genres = load_genres()
    genre_of = {imap[m]: gs for m, gs in genres.items() if m in imap}

    rows = []

    def evaluate(score_fn, label):
        """Score every eval user with score_fn, average HR@10/NDCG@10.

        Appends (label, mean_hr, mean_ndcg) to the enclosing `rows` list
        and echoes the line to stderr for live progress.
        """
        hrs, ndcgs = [], []
        for u in eval_users:
            scores = score_fn(u)
            top10 = np.argpartition(-scores, 10)[:10]
            top10 = top10[np.argsort(-scores[top10])]
            hr, nd = hr_ndcg_at10(top10, relevant_by_user[u])
            hrs.append(hr)
            ndcgs.append(nd)
        rows.append((label, float(np.mean(hrs)), float(np.mean(ndcgs))))
        print(f"  {label:<42} HR@10={rows[-1][1]:.4f}  NDCG@10={rows[-1][2]:.4f}",
              file=sys.stderr)

    def pop_scores(u):
        """Baseline (a): rank candidates by global train-set rating count.

        The user's already-rated movies are excluded (-inf), matching how
        every other method is scored.
        """
        s = pop.copy()
        s[list(train_by_user[u])] = -np.inf
        return s

    def genre_pop_scores(u):
        """Baseline (b): popularity restricted to the user's top-3 genres.

        The user's genres are ranked by rating-weighted counts over their
        train ratings (mirroring the dashboard's top-3-genre inference);
        any movie tagged with one of those genres outranks all others
        (+1e6 offset), popularity ordering within each block. Rated
        movies excluded.
        """
        weight = Counter()
        for j, r in train_by_user[u].items():
            for g in genre_of.get(j, []):
                weight[g] += r
        top3 = {g for g, _ in weight.most_common(3)}
        s = pop.copy()
        in_genre = np.array([bool(top3 & set(genre_of.get(j, []))) for j in range(n_items)])
        s = np.where(in_genre, s + 1e6, s)   # top-3-genre movies rank first, by popularity
        s[list(train_by_user[u])] = -np.inf
        return s

    print("\nbaselines:", file=sys.stderr)
    evaluate(pop_scores, "popularity")
    evaluate(genre_pop_scores, "genre-matched popularity")

    # ---- CF grid ----
    from cf import train_item_item
    user_means = {u: np.mean(list(rs.values())) for u, rs in train_by_user.items()}
    for beta in BETAS:
        print(f"\ntraining CF (beta={beta})...", file=sys.stderr)
        nbr_idx, nbr_sim = train_item_item(X, beta=beta)
        for norm in (True, False):
            evaluate(
                lambda u, ni=nbr_idx, ns=nbr_sim, nm=norm: fold_in_scores(
                    ni, ns, train_by_user[u], user_means[u], n_items, normalize=nm),
                f"item-item CF beta={beta} {'normalized' if norm else 'unnormalized'}",
            )

    # ---- markdown table ----
    print("\n| Method | HR@10 | NDCG@10 |")
    print("|---|---|---|")
    for label, hr, nd in rows:
        print(f"| {label} | {hr:.4f} | {nd:.4f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
