/**
 * BM25 over the search children, built in-browser at first search.
 *
 * Constants per house spec: k1=1.2, b=0.75, and the Lucene non-negative
 * IDF: ln(1 + (N - df + 0.5)/(df + 0.5)) — never negative for terms that
 * appear in more than half the corpus.
 */

import type { Child, LegHit } from "./types";

const K1 = 1.2;
const B = 0.75;

/** Inverted index + stats needed to score at query time. */
export interface Bm25Index {
  /** term -> postings as parallel [childId[], termFreq[]] arrays. */
  postings: Map<string, { ids: number[]; tfs: number[] }>;
  /** Token count per child. */
  docLen: Float32Array;
  /** Mean token count across children. */
  avgLen: number;
  /** Corpus size N (number of children). */
  n: number;
}

/**
 * Tokenize text for BM25: lowercase, alphanumeric runs only.
 *
 * Deliberately no stemming/stopwords — IDF already damps common terms,
 * and matching the build side exactly matters more than linguistic
 * cleverness (documented simplification).
 *
 * @param text - raw text (query or child).
 * @returns Ordered token array (duplicates preserved).
 */
export function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

/**
 * Build the inverted index over all children (~59k docs, <1s).
 *
 * @param children - the full child array (index = child id).
 * @returns Ready-to-score Bm25Index.
 */
export function buildBm25(children: Child[]): Bm25Index {
  const postings = new Map<string, { ids: number[]; tfs: number[] }>();
  const docLen = new Float32Array(children.length);
  let total = 0;

  children.forEach((child, id) => {
    const tokens = tokenize(child.text);
    docLen[id] = tokens.length;
    total += tokens.length;
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const [term, f] of tf) {
      let p = postings.get(term);
      if (!p) postings.set(term, (p = { ids: [], tfs: [] }));
      p.ids.push(id);
      p.tfs.push(f);
    }
  });

  return { postings, docLen, avgLen: total / children.length, n: children.length };
}

/**
 * Score a query against the index; return the top-k children.
 *
 * Standard BM25 accumulation over query terms (deduped — repeating a
 * term in a 3-word query shouldn't double its weight).
 *
 * @param index - built Bm25Index.
 * @param query - raw user query.
 * @param topK - number of hits to return.
 * @returns Hits sorted by score desc; empty if no term matches.
 */
export function searchBm25(index: Bm25Index, query: string, topK: number): LegHit[] {
  const scores = new Map<number, number>();
  for (const term of new Set(tokenize(query))) {
    const p = index.postings.get(term);
    if (!p) continue;
    const idf = Math.log(1 + (index.n - p.ids.length + 0.5) / (p.ids.length + 0.5));
    for (let j = 0; j < p.ids.length; j++) {
      const id = p.ids[j];
      const tf = p.tfs[j];
      const norm = tf / (tf + K1 * (1 - B + (B * index.docLen[id]) / index.avgLen));
      scores.set(id, (scores.get(id) ?? 0) + idf * (K1 + 1) * norm);
    }
  }
  return [...scores.entries()]
    .map(([childId, score]) => ({ childId, score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
