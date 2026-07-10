/**
 * Search orchestrator: runs both legs, fuses, groups, reranks, reports
 * mode. Rerank failures degrade to fusion order (never a dead search).
 */

import { buildBm25, searchBm25, type Bm25Index } from "./bm25";
import { LEG_TOP_K, RERANK_MIN_LOGIT } from "./config";
import { byId, loadCatalog, loadChildren } from "./data";
import { searchDense } from "./dense";
import { groupToParents, rrfFuse } from "./fuse";
import type { Child, Movie, SearchResult } from "./types";

/** Which retrieval mode actually ran (drives the UI badge). */
export type SearchMode = "hybrid" | "bm25-only";

/** A completed search: results plus how they were produced. */
export interface SearchOutcome {
  results: SearchResult[];
  mode: SearchMode;
}

let bm25Index: Bm25Index | null = null;
let childrenCache: Child[] | null = null;
let movieMap: Map<number, Movie> | null = null;

/**
 * One-time lazy init: load children + catalog, build BM25 in memory.
 * Subsequent searches reuse everything.
 */
async function ensureReady(): Promise<void> {
  if (bm25Index) return;
  const [children, catalog] = await Promise.all([loadChildren(), loadCatalog()]);
  childrenCache = children;
  movieMap = byId(catalog);
  bm25Index = buildBm25(children);
}

/**
 * Run a full hybrid search for a query.
 *
 * Both legs run concurrently: dense = embed in-browser then Worker->
 * Pinecone; BM25 = local. If the dense leg fails for any reason (Worker
 * down, offline, cold index) the search degrades to lexical-only and
 * says so via mode — it never throws for that case.
 *
 * @param query - raw user text.
 * @returns Top-20 grouped results (grid shows the first 9) + mode flag.
 */
export async function search(query: string): Promise<SearchOutcome> {
  await ensureReady();
  const children = childrenCache!;
  const movies = movieMap!;

  const bm25Hits = searchBm25(bm25Index!, query, LEG_TOP_K);
  let denseHits: Awaited<ReturnType<typeof searchDense>> = [];
  let mode: SearchMode = "hybrid";
  try {
    // dynamic import keeps transformers.js (~800 kB) out of the entry chunk
    const { embedQuery } = await import("./embed");
    denseHits = await searchDense(await embedQuery(query));
  } catch (e) {
    console.warn("dense leg unavailable, degrading to BM25-only:", e);
    mode = "bm25-only";
  }

  let results = groupToParents(rrfFuse(denseHits, bm25Hits), children, movies);

  try {
    // dynamic import: the cross-encoder (~23 MB) loads on first use only
    const { rerank, buildDocParts } = await import("./rerank");
    docPartsCache ??= buildDocParts(children);
    results = await rerank(query, results, docPartsCache);
    // no-answer gate: the cross-encoder is the only calibrated confidence
    // signal in the pipeline — drop what it calls irrelevant; an empty
    // survivor list renders the sad-potato state instead of a grid of
    // confidently-presented noise (dense ALWAYS returns nearest vectors,
    // even for gibberish).
    results = results.filter(
      (r) => r.rerankScore !== undefined && r.rerankScore >= RERANK_MIN_LOGIT,
    );
  } catch (e) {
    console.warn("rerank unavailable, keeping fusion order:", e);
  }

  return { results, mode };
}

let docPartsCache: Map<number, string> | null = null;
