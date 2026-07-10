/**
 * Blend mode — the ONE bridge between retrieval (content-based) and the
 * recommender (item-item CF). Applied AFTER rerank, re-ordering ONLY the
 * ~20 search candidates; never injects CF-only movies (the query decides
 * relevance, CF only nudges order).
 *
 *   final = (1−α)·sigmoid(rerank) + α·sigmoid(cf/τ)
 *
 * Convex combination per Bruch, Gai & Ingber (TOIS 2023): the robust,
 * single-parameter fusion when you can't train a ranker. Sigmoid (not
 * min-max) → fixed scale and cf=0 → 0.5 = NEUTRAL: a CF-silent candidate
 * (no rated neighbors) is never penalized. Pre-onboarding, α is forced
 * to 0 (pure search).
 */

import { dampedMean } from "./foldin";
import type { NeighborMap } from "./neighbors";
import type { Ratings } from "./ratings";
import type { SearchResult } from "./types";

/**
 * Temperature for the CF sigmoid. Calibrated offline against a ~14-
 * rating persona (scripts: p90 |cf| ≈ 0.02, max ≈ 0.17): τ=0.05 maps a
 * strong fold-in score (0.1) to sigmoid(2) ≈ 0.88 and noise-level
 * scores to ~0.5. Spread grows with ratings count — the α slider
 * compensates; revisit if typical users rate 100+.
 */
export const TAU = 0.05;

/**
 * Temperature for the RELEVANCE sigmoid. ms-marco logits span roughly
 * ±10; raw sigmoid saturates every good result at ≈1.0, erasing
 * relevance differences so even a small α let taste take over (caught
 * live: α=0.15 ordered identically to α=1). τ_rel=2.5 keeps the top-20
 * spread inside sigmoid's active range.
 */
export const REL_TAU = 2.5;

/** Default personal-nudge weight (subtle by design). */
export const DEFAULT_ALPHA = 0.15;

/** Per-result CF context the blend adds for provenance. */
export interface Nudge {
  /** Unnormalized fold-in score for this candidate (0 = silent). */
  cf: number;
  /** ❤️-rated movies that pushed this up, strongest first (≤2). */
  because: string[];
}

/**
 * Compute each candidate's fold-in score + provenance from the user's
 * ratings — candidate-centric, over the candidate's own neighbor list.
 *
 * @param results - the reranked top-N search results.
 * @param neighbors - the CF model.
 * @param ratings - the user's ❤️/🥔 map.
 * @param titleOf - movieId -> display title (for "because you ❤️ X").
 * @returns movieId -> Nudge (absent = CF-silent or the movie is rated).
 */
export function cfNudges(
  results: SearchResult[],
  neighbors: NeighborMap,
  ratings: Ratings,
  titleOf: Map<number, string>,
): Map<number, Nudge> {
  const mean = dampedMean(ratings);
  const out = new Map<number, Nudge>();
  for (const r of results) {
    if (ratings[r.movie.id] !== undefined) continue; // rated: stays neutral
    const nbrs = neighbors.get(r.movie.id);
    if (!nbrs) continue;
    let cf = 0;
    const contributors: [number, number][] = [];
    for (const [nid, sim] of nbrs) {
      const rating = ratings[nid];
      if (rating === undefined) continue;
      const c = sim * (rating - mean);
      cf += c;
      if (c > 0 && rating > 0) contributors.push([nid, c]);
    }
    if (cf === 0) continue;
    contributors.sort((a, b) => b[1] - a[1]);
    out.set(r.movie.id, {
      cf,
      because: contributors.slice(0, 2)
        .map(([id]) => titleOf.get(id))
        .filter((t): t is string => !!t),
    });
  }
  return out;
}

/**
 * Logistic squash to (0,1).
 *
 * @param x - any real.
 * @returns sigmoid(x); 0 maps to exactly 0.5 (the neutral point).
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Re-order reranked results by the convex blend of relevance and taste.
 *
 * Candidates without a rerankScore (cross-encoder failed → fusion order)
 * fall back to a rank-based relevance proxy so blending still works.
 * CF-silent candidates use exactly 0.5 — neither boosted nor punished.
 *
 * @param results - reranked results (order = pure relevance).
 * @param nudges - cfNudges() output.
 * @param alpha - taste weight in [0,1]; 0 returns relevance order.
 * @returns New array sorted by blended score (input untouched).
 */
export function applyBlend(
  results: SearchResult[],
  nudges: Map<number, Nudge>,
  alpha: number,
): SearchResult[] {
  if (alpha <= 0) return [...results];
  return results
    .map((r, i) => {
      const relevance =
        r.rerankScore !== undefined
          ? sigmoid(r.rerankScore / REL_TAU)
          : 1 - i / Math.max(results.length, 1); // rank proxy, already (0,1]
      const cf = nudges.get(r.movie.id)?.cf ?? 0;
      const taste = cf === 0 ? 0.5 : sigmoid(cf / TAU);
      return { r, final: (1 - alpha) * relevance + alpha * taste };
    })
    .sort((a, b) => b.final - a.final)
    .map((x) => x.r);
}
