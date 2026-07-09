# EVALUATION.md

Honest offline evaluation, updated as each stage lands. Search-side eval
(golden queries, per-leg recall/MRR) arrives with step 3–4.

## 1. Recommender: item-item CF vs popularity

**Question:** does item-item collaborative filtering (Sarwar et al. 2001)
actually beat "just recommend popular movies" on this dataset? Most demo
recommenders never check.

### Protocol

- **Data:** MovieLens `ml-latest-small` — 100,836 ratings, 610 users, 9,724
  movies.
- **Split:** temporal, per user — earliest 80% of each user's ratings train
  (80,672), latest 20% test (20,164). No test information touches training.
- **Relevant** = held-out rating ≥ 4.0. Users with ≥ 1 relevant held-out item
  are evaluated: **591 of 610**. 619 relevant test ratings are on movies
  absent from the training split — unreachable by *any* method scored here;
  they depress all numbers equally.
- **Metrics:** hit-rate@10 (≥ 1 relevant item in top-10) and NDCG@10 (binary
  relevance), averaged over evaluated users. Rated-in-train movies are
  excluded from each user's candidate ranking.
- **Model:** adjusted cosine (user-mean-centered) item-item similarities,
  shrunk by n/(n+β) where n = co-rater count; top-50 neighbors per movie.
- **Fold-in scoring** (what the runtime does): for candidate *j*,
  score(j) = Σ sim(i,j) · (r_ui − r̄_u) over the user's rated movies *i* in
  *j*'s neighbor list. Both the *normalized* variant (÷ Σ|sim|, i.e. a rating
  prediction) and the *unnormalized* sum were tested.

### Results

| Method | HR@10 | NDCG@10 |
|---|---|---|
| popularity | 0.3215 | 0.0746 |
| genre-matched popularity (top-3 user genres) | 0.3367 | 0.0699 |
| item-item CF, normalized, β=0 | 0.0423 | 0.0061 |
| item-item CF, normalized, β=5 | 0.0102 | 0.0010 |
| item-item CF, normalized, β=25 | 0.0102 | 0.0018 |
| item-item CF, normalized, β=100 | 0.0102 | 0.0015 |
| item-item CF, unnormalized, β=0 | 0.3266 | 0.0730 |
| item-item CF, unnormalized, β=5 | 0.4010 | 0.1007 |
| item-item CF, unnormalized, β=25 | 0.3976 | 0.1084 |
| item-item CF, unnormalized, β=50 | 0.4112 | 0.1096 |
| item-item CF, unnormalized, β=100 | 0.4112 | 0.1099 |
| item-item CF, unnormalized, β=200 | 0.4162 | 0.1105 |
| **item-item CF, unnormalized, β=400 (shipped)** | **0.4247** | **0.1125** |
| item-item CF, unnormalized, β=800 | 0.4213 | 0.1114 |
| item-item CF, unnormalized, β=3200 | 0.4162 | 0.1095 |

### Findings

1. **CF beats popularity.** +32% relative HR@10 (0.42 vs 0.32), +51% relative
   NDCG@10 (0.113 vs 0.075) over the popularity baseline. The credibility
   check passes on this dataset.
2. **The normalized fold-in is catastrophic for top-N ranking** (HR@10
   collapses to ~0.01). Dividing by Σ|sim| converts the score into a rating
   *prediction*: a candidate with one weak neighbor rated 5★ ties or beats a
   candidate supported by forty strong neighbors, so low-evidence items flood
   the top-10. This is a known top-N pitfall; the eval caught it in our own
   spec. **The runtime ranks with the unnormalized sum.**
3. **Shrinkage matters and wants to be large.** β in the hundreds beats the
   folk-default single digits: with 610 users, cosine over few co-raters is
   noise, and n/(n+β) with large β effectively demands strong co-rating
   support. The curve is a broad plateau over β ≈ 100–800; differences inside
   it (±0.02 HR at n=591) are within noise. β=400 is chosen mid-plateau.
   *Honesty note:* β was scanned on the same single split reported here —
   light test-set tuning; treat the plateau, not the point, as the result.
4. **Genre-matched popularity** edges plain popularity on HR@10 but is worse
   on NDCG@10 — restricting to a user's top genres finds *a* hit slightly
   more often but orders the list worse.

### Shipped model

Retrained on **all** 100,836 ratings (the eval split exists only for
measurement): β=400, top-50 neighbors, unnormalized fold-in at runtime.
`neighbors.json` = 7.25 MB minified (9,722 movies × ≤50 [movieId, sim]
pairs) — above the spec's 2–3 MB guess, within the ≤20 MB total payload
budget. Qualitative check: Toy Story's top neighbors are Aladdin / Toy
Story 2 / The Lion King; Pulp Fiction's are The Usual Suspects / Se7en /
Fight Club — the "because you rated X, Y" explanations will be sane.

### Known limitations

- **610 users.** Similarities are noisy for niche movies regardless of
  shrinkage; the documented upgrade path is `ml-25m` (+ tag genome).
- Single temporal split, no confidence intervals; the β plateau (not the
  exact peak) is the defensible claim.
- Binary relevance at the ≥ 4.0 threshold; results not re-checked at 3.5/4.5.
- Live users rate ❤️/🥔 (±1), not 5-star: the runtime uses a damped user
  mean (ū = Σr/(n+β), β≈5) so one-sided raters keep a signal. The offline
  eval uses true 5-star means — this gap is real and unmeasured offline.

## 2. Search: hybrid retrieval (pending — step 3/4)

Golden set of ~50 hand-written queries (exact-title / actor-ish / vibe /
constraint), graded relevant movies, recall@9 and MRR per leg: dense-only,
BM25-only, fused (RRF k=60), fused + cross-encoder rerank. Optional
convex-combination fusion row (Bruch et al. 2023). Blend mode is
deliberately **not** evaluated against the golden set — it perturbs query
relevance by design; the debug panel's side-by-side is its demonstration.

---

*Citations: Sarwar et al. 2001 (item-based CF); Cormack et al. 2009 (RRF);
Bruch, Gai & Ingber 2023 (fusion functions); Harper & Konstan 2015
(MovieLens).*
