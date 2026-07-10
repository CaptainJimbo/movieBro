/** Runtime configuration constants (retrieval knobs match the spec). */

/**
 * Dense-search proxy endpoint (Cloudflare Worker in front of Pinecone).
 * Overridable per-environment via VITE_WORKER_URL; the fallback is the
 * deployed workers.dev URL.
 */
export const WORKER_URL: string =
  (import.meta.env.VITE_WORKER_URL as string | undefined) ??
  "https://moviebro-search.captainjimbo.workers.dev";

/** RRF constant k (Cormack et al. 2009; house standard). */
export const RRF_K = 60;

/** Candidates fetched per retrieval leg before fusion. */
export const LEG_TOP_K = 60;

/** Parents kept after best-hit grouping (rerank pool, step 4). */
export const GROUP_TOP_N = 20;

/** Results displayed in the grid (3x3). */
export const GRID_SIZE = 9;

/** bge-small query prefix — queries get it, passages never do. */
export const BGE_QUERY_PREFIX =
  "Represent this sentence for searching relevant passages: ";

/** TMDB poster CDN base (w342 = grid-sized). */
export const POSTER_BASE = "https://image.tmdb.org/t/p/w342";
