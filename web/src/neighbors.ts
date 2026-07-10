/**
 * Loader for the item-item CF model (neighbors.json, ~7 MB) — the
 * build-time artifact from scripts/train_cf.py. Lazy-loaded when the
 * dashboard first renders, then cached.
 */

/** movieId -> its top-K CF neighbors as [neighborId, similarity] pairs. */
export type NeighborMap = Map<number, [number, number][]>;

let neighborsPromise: Promise<NeighborMap> | null = null;

/**
 * Fetch and cache the neighbor lists.
 *
 * @returns Map of movieId -> [[neighborId, sim], ...] (sims are the
 *   shrunk adjusted-cosine values, strongest |sim| first).
 */
export function loadNeighbors(): Promise<NeighborMap> {
  neighborsPromise ??= fetch(`${import.meta.env.BASE_URL}data/neighbors.json`)
    .then((r) => {
      if (!r.ok) throw new Error(`neighbors fetch ${r.status}`);
      return r.json() as Promise<{ neighbors: Record<string, [number, number][]> }>;
    })
    .then(({ neighbors }) => {
      const map: NeighborMap = new Map();
      for (const [mid, list] of Object.entries(neighbors)) map.set(Number(mid), list);
      return map;
    });
  return neighborsPromise;
}
