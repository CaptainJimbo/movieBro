/**
 * Live fold-in: score every candidate movie from the user's ❤️/🥔
 * ratings over the FIXED item-item similarities (neighbors.json never
 * retrains on-device; more ratings sharpen personalization only).
 *
 * Two eval-driven rules (EVALUATION.md §1):
 *  - UNNORMALIZED sum — dividing by Σ|sim| collapses top-N ranking
 *    (HR@10 0.42 → 0.01, finding 2).
 *  - DAMPED user mean ū = Σr/(n+β), β=5 — with the ±1 scale a
 *    hearts-only user would otherwise have (r − ū) = 0 everywhere and
 *    the fold-in would go silent; damping keeps signal while still
 *    removing generosity bias as evidence accumulates.
 */

import type { NeighborMap } from "./neighbors";
import type { Movie } from "./types";
import type { Ratings } from "./ratings";

const MEAN_DAMPING_BETA = 5;

/** One scored recommendation with its provenance. */
export interface Pick {
  /** The recommended movie. */
  movie: Movie;
  /** Unnormalized fold-in score (blend consumes this at step 6). */
  score: number;
  /** Rated movies that pushed this pick up, strongest first (❤️ only). */
  because: Movie[];
}

/**
 * Damped mean of the user's ratings: ū = Σr / (n + β).
 *
 * @param ratings - the ❤️/🥔 map.
 * @returns ū in (−1, 1); 0 when unrated.
 */
export function dampedMean(ratings: Ratings): number {
  const values = Object.values(ratings);
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / (values.length + MEAN_DAMPING_BETA);
}

/**
 * Score ALL unrated movies by folding the user's ratings into the
 * neighbor model — the exact runtime counterpart of the eval's
 * fold_in_scores(normalize=False).
 *
 * For each candidate, walk its OWN neighbor list; for every neighbor
 * the user rated, accumulate sim × (rating − ū). O(movies × K) ≈ 500k
 * ops — instant in JS.
 *
 * @param movies - full catalog.
 * @param neighbors - the CF model.
 * @param ratings - the user's ❤️/🥔 map.
 * @returns movieId -> {score, contributors} for candidates with ≥1 rated
 *   neighbor (rated movies excluded).
 */
export function foldIn(
  movies: Movie[],
  neighbors: NeighborMap,
  ratings: Ratings,
): Map<number, { score: number; contributors: [number, number][] }> {
  const mean = dampedMean(ratings);
  const out = new Map<number, { score: number; contributors: [number, number][] }>();

  for (const m of movies) {
    if (ratings[m.id] !== undefined) continue; // never recommend rated
    const nbrs = neighbors.get(m.id);
    if (!nbrs) continue;
    let score = 0;
    const contributors: [number, number][] = []; // [ratedMovieId, contribution]
    for (const [nid, sim] of nbrs) {
      const r = ratings[nid];
      if (r === undefined) continue;
      const contribution = sim * (r - mean);
      score += contribution;
      if (contribution > 0 && r > 0) contributors.push([nid, contribution]);
    }
    if (contributors.length > 0 || score !== 0) {
      contributors.sort((a, b) => b[1] - a[1]);
      out.set(m.id, { score, contributors });
    }
  }
  return out;
}

/**
 * The user's top-3 genres by rating-weighted counts: each rated movie
 * adds its rating (±1) to each of its genres, so 🥔s actively push a
 * genre down. Mirrors the eval's genre-matched baseline construction.
 *
 * @param movies - full catalog (id lookup).
 * @param ratings - the ❤️/🥔 map.
 * @returns Up to 3 genre names, best first (only positive-sum genres).
 */
export function topGenres(movies: Map<number, Movie>, ratings: Ratings): string[] {
  const weight = new Map<string, number>();
  for (const [mid, r] of Object.entries(ratings)) {
    const m = movies.get(Number(mid));
    for (const g of m?.genres ?? []) {
      if (g === "(no genres listed)") continue;
      weight.set(g, (weight.get(g) ?? 0) + r);
    }
  }
  return [...weight.entries()]
    .filter(([, w]) => w > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([g]) => g);
}

/**
 * The dashboard picks: one best fold-in candidate per top genre, each
 * with its "because you ❤️ X, Y" provenance (top 2 contributors).
 *
 * @param movies - full catalog.
 * @param neighbors - the CF model.
 * @param ratings - the user's ❤️/🥔 map.
 * @returns [genre, Pick] pairs (≤3; a movie appears under one genre only).
 */
export function genrePicks(
  movies: Movie[],
  neighbors: NeighborMap,
  ratings: Ratings,
): [string, Pick][] {
  const byId = new Map(movies.map((m) => [m.id, m]));
  const scores = foldIn(movies, neighbors, ratings);
  const genres = topGenres(byId, ratings);

  const ranked = [...scores.entries()].sort((a, b) => b[1].score - a[1].score);
  const used = new Set<number>();
  const picks: [string, Pick][] = [];
  for (const genre of genres) {
    for (const [mid, s] of ranked) {
      const m = byId.get(mid);
      if (!m || used.has(mid) || !m.genres.includes(genre) || s.score <= 0) continue;
      used.add(mid);
      picks.push([genre, {
        movie: m,
        score: s.score,
        because: s.contributors.slice(0, 2)
          .map(([id]) => byId.get(id))
          .filter((x): x is Movie => !!x),
      }]);
      break;
    }
  }
  return picks;
}
