/**
 * Rank fusion + parent grouping — the heart of the hybrid retrieval.
 *
 * RRF (Cormack et al. 2009) with k=60 fuses the two legs' RANKS (scores
 * on incompatible scales are never compared); best-hit grouping then
 * takes each movie's MAX child score — never a sum, so a movie with many
 * mediocre children can't beat one perfect match.
 */

import { GROUP_TOP_N, RRF_K } from "./config";
import type { Child, LegHit, Movie, SearchResult } from "./types";

/** Fused child score plus which legs ranked it. */
interface FusedChild {
  childId: number;
  score: number;
  dense: boolean;
  bm25: boolean;
}

/**
 * Reciprocal-rank-fuse the two legs at child level.
 *
 * score(c) = Σ_legs 1/(k + rank_leg(c)), rank starting at 1. A child
 * ranked by both legs gets both contributions — the agreement bonus is
 * what makes RRF work.
 *
 * @param dense - dense-leg hits (rank order = array order).
 * @param bm25 - BM25-leg hits (rank order = array order).
 * @returns All seen children with fused scores, sorted desc.
 */
export function rrfFuse(dense: LegHit[], bm25: LegHit[]): FusedChild[] {
  const fused = new Map<number, FusedChild>();
  const add = (hits: LegHit[], leg: "dense" | "bm25") => {
    hits.forEach((h, i) => {
      let f = fused.get(h.childId);
      if (!f) fused.set(h.childId, (f = { childId: h.childId, score: 0, dense: false, bm25: false }));
      f.score += 1 / (RRF_K + i + 1);
      f[leg] = true;
    });
  };
  add(dense, "dense");
  add(bm25, "bm25");
  return [...fused.values()].sort((a, b) => b.score - a.score);
}

/**
 * Group fused children to parent movies by best hit (max, never sum),
 * keeping the top N parents.
 *
 * @param fused - output of rrfFuse().
 * @param children - full child array (childId -> parent resolution).
 * @param movies - movieId -> Movie lookup.
 * @returns Top-N SearchResults, each carrying its best child + leg
 *   badges for the provenance hover.
 */
export function groupToParents(
  fused: FusedChild[],
  children: Child[],
  movies: Map<number, Movie>,
): SearchResult[] {
  const best = new Map<number, FusedChild>();
  for (const f of fused) {
    const movieId = children[f.childId].movieId;
    if (!best.has(movieId)) best.set(movieId, f); // fused is sorted: first = max
  }
  return [...best.entries()]
    .map(([movieId, f]) => {
      const movie = movies.get(movieId);
      return movie
        ? {
            movie,
            score: f.score,
            bestChild: children[f.childId],
            legs: { dense: f.dense, bm25: f.bm25 },
          }
        : null;
    })
    .filter((r): r is SearchResult => r !== null)
    .sort((a, b) => b.score - a.score)
    .slice(0, GROUP_TOP_N);
}
