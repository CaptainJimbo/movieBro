/**
 * In-browser cross-encoder rerank (step 4 of the pipeline).
 *
 * ms-marco-MiniLM-L-6-v2 scores (query, document) PAIRS jointly — unlike
 * the bi-encoder dense leg, the model attends across both texts, which
 * is why it's too expensive to run over the corpus but excellent for
 * re-ordering the top-20 fusion candidates. Document = "Title (year).
 * <overview sentences>" reconstructed from the movie's overview children.
 */

import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

import type { Child, SearchResult } from "./types";

const MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2";

let modelPromise: Promise<{
  tokenizer: PreTrainedTokenizer;
  model: PreTrainedModel;
}> | null = null;

/**
 * Lazily load (and cache) the cross-encoder tokenizer + model
 * (~23 MB quantized, one-time download).
 *
 * @returns The ready pair; subsequent calls reuse the same promise.
 */
function getModel() {
  modelPromise ??= Promise.all([
    AutoTokenizer.from_pretrained(MODEL_ID),
    AutoModelForSequenceClassification.from_pretrained(MODEL_ID),
  ]).then(([tokenizer, model]) => ({ tokenizer, model }));
  return modelPromise;
}

/**
 * Build the rerank document for one movie: title, year, overview, cast
 * and keywords. The document must carry the SAME evidence the retrieval
 * legs matched on — the eval showed a title+overview-only doc makes the
 * cross-encoder DEMOTE actor/constraint hits it can't see (recall@9
 * 0.45 -> 0.40; with full evidence rerank lifts MRR@9 0.50 -> 0.60,
 * EVALUATION.md section 2).
 *
 * @param result - a grouped search result.
 * @param docParts - movieId -> joined evidence text (see buildDocParts).
 * @returns Document text for the (query, doc) pair.
 */
function docFor(result: SearchResult, docParts: Map<number, string>): string {
  const m = result.movie;
  return `${m.title}${m.year ? ` (${m.year})` : ""}. ${docParts.get(m.id) ?? ""}`;
}

/**
 * Precompute movieId -> rerank evidence text from the children array:
 * overview sentences (stored in order, so joining reconstructs the TMDB
 * overview) followed by the cast and keywords children.
 *
 * @param children - full child array.
 * @returns Map used by docFor(); movies without any such children are
 *   absent (their doc is title-only).
 */
export function buildDocParts(children: Child[]): Map<number, string> {
  const out = new Map<number, string>();
  for (const c of children) {
    if (c.type !== "overview" && c.type !== "cast" && c.type !== "keywords") continue;
    out.set(c.movieId, out.has(c.movieId) ? `${out.get(c.movieId)} ${c.text}` : c.text);
  }
  return out;
}

/**
 * Rerank grouped results with the cross-encoder.
 *
 * All (query, doc) pairs are scored in ONE batched forward pass; results
 * come back sorted by logit desc with `rerankScore` attached (the blend
 * stage consumes it at step 6).
 *
 * @param query - raw user query.
 * @param results - top-N grouped results from fusion.
 * @param docParts - movieId -> evidence text map (buildDocParts()).
 * @returns Re-ordered copy of results.
 * @throws Whatever model load/inference throws — the caller degrades to
 *   fusion order.
 */
export async function rerank(
  query: string,
  results: SearchResult[],
  docParts: Map<number, string>,
): Promise<SearchResult[]> {
  if (results.length === 0) return results;
  const { tokenizer, model } = await getModel();

  const docs = results.map((r) => docFor(r, docParts));
  const inputs = tokenizer(new Array(docs.length).fill(query), {
    text_pair: docs,
    padding: true,
    truncation: true,
  });
  const { logits } = await model(inputs);
  const scores = Array.from(logits.data as Float32Array);

  return results
    .map((r, i) => ({ ...r, rerankScore: scores[i] }))
    .sort((a, b) => (b.rerankScore ?? 0) - (a.rerankScore ?? 0));
}
