#!/usr/bin/env python3
"""Train the shipping item-item model on ALL ratings -> data/build/neighbors.json.

beta=400 sits mid-plateau on the temporal-split eval (see EVALUATION.md);
scores are ranked with the UNNORMALIZED fold-in (sum of sim * centered rating)
— the normalized variant collapses top-N quality (also EVALUATION.md).
"""

import json
import sys
from pathlib import Path

import numpy as np

from cf import TOP_K, build_matrix, load_ratings, train_item_item

BETA = 400
OUT = Path(__file__).resolve().parent.parent / "data" / "build"


def main() -> int:
    """Train the shipping neighbor model and write neighbors.json.

    Unlike the eval (which trains on the 80% temporal split), this trains
    on ALL ratings — the split exists only for measurement. Output format:
    {"model", "beta", "k", "scoring", "neighbors": {movieId: [[neighborId,
    sim], ...]}} with sims rounded to 4 decimals, minified. Prints
    coverage/size stats to stderr.

    Returns:
        Process exit code (0 on success).
    """
    users, items, ratings, _ = load_ratings()
    X, _, iids = build_matrix(users, items, ratings)
    print(f"training on all {len(ratings)} ratings | "
          f"{X.shape[0]} users x {X.shape[1]} items | beta={BETA}", file=sys.stderr)

    nbr_idx, nbr_sim = train_item_item(X, beta=BETA)

    neighbors = {}
    for i in range(X.shape[1]):
        valid = nbr_idx[i] >= 0
        if not valid.any():
            continue
        neighbors[int(iids[i])] = [
            [int(iids[j]), round(float(s), 4)]
            for j, s in zip(nbr_idx[i][valid], nbr_sim[i][valid])
            if s != 0.0
        ]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "neighbors.json"
    path.write_text(json.dumps(
        {"model": "item-item adjusted cosine", "beta": BETA, "k": TOP_K,
         "scoring": "unnormalized fold-in", "neighbors": neighbors},
        separators=(",", ":")))

    sizes = [len(v) for v in neighbors.values()]
    print(f"movies with neighbors: {len(neighbors)}", file=sys.stderr)
    print(f"avg/median neighbors : {np.mean(sizes):.1f} / {int(np.median(sizes))}",
          file=sys.stderr)
    print(f"file size            : {path.stat().st_size / 1e6:.2f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
