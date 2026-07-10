/** Loading + caching of the shipped data artifacts. */

import type { Child, Movie } from "./types";

let catalogPromise: Promise<Movie[]> | null = null;
let childrenPromise: Promise<Child[]> | null = null;

/**
 * Fetch and cache the movie catalog (movies.json, ~1.4 MB).
 *
 * @returns All catalog movies; subsequent calls reuse the same promise.
 */
export function loadCatalog(): Promise<Movie[]> {
  catalogPromise ??= fetch(`${import.meta.env.BASE_URL}data/movies.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`catalog fetch ${r.status}`);
      return r.json() as Promise<Movie[]>;
    });
  return catalogPromise;
}

/**
 * Fetch and cache the search children (search-index.json, ~6.7 MB) —
 * lazy-loaded on first search per the performance budget, not at boot.
 *
 * @returns Children in id order (array index = child id = Pinecone id).
 */
export function loadChildren(): Promise<Child[]> {
  childrenPromise ??= fetch(`${import.meta.env.BASE_URL}data/search-index.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`search index fetch ${r.status}`);
      return r.json() as Promise<{ children: [number, string, string][] }>;
    })
    .then(({ children }) =>
      children.map(([movieId, type, text]) => ({ movieId, type, text }) as Child),
    );
  return childrenPromise;
}

/**
 * Build a movieId -> Movie lookup from the catalog.
 *
 * @param movies - catalog array from loadCatalog().
 * @returns Map for O(1) id resolution during result assembly.
 */
export function byId(movies: Movie[]): Map<number, Movie> {
  return new Map(movies.map((m) => [m.id, m]));
}
