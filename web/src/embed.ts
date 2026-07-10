/**
 * In-browser query embedding via transformers.js (bge-small-en-v1.5).
 *
 * The bge asymmetry: QUERIES are prefixed with BGE_QUERY_PREFIX; passages
 * were embedded prefix-free at build time. Pooling is CLS (what bge
 * trains with), normalized so cosine == dot product.
 */

import { pipeline, type FeatureExtractionPipeline, type ProgressInfo } from "@huggingface/transformers";

import { BGE_QUERY_PREFIX } from "./config";

/**
 * pipeline() retyped for our single use — its native overload union is
 * too complex for tsc (TS2590) when passed an options object.
 */
const makePipeline = pipeline as (
  task: "feature-extraction",
  model: string,
  options?: { progress_callback?: (p: ProgressInfo) => void },
) => Promise<FeatureExtractionPipeline>;

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

/**
 * Lazily create (and cache) the feature-extraction pipeline. First call
 * downloads the quantized model (~35 MB) — callers should surface
 * progress UI around it.
 *
 * @param onProgress - optional coarse progress callback (0..1) for the
 *   model download phase.
 * @returns The ready pipeline; subsequent calls reuse it.
 */
export function getExtractor(
  onProgress?: (frac: number) => void,
): Promise<FeatureExtractionPipeline> {
  extractorPromise ??= makePipeline("feature-extraction", "Xenova/bge-small-en-v1.5", {
    progress_callback: (p: ProgressInfo) => {
      if (onProgress && p.status === "progress") onProgress(p.progress / 100);
    },
  });
  return extractorPromise;
}

/**
 * Embed one search query into a normalized 384-dim vector.
 *
 * @param query - raw user query text (prefix applied here).
 * @returns Plain number array ready to POST to the Worker.
 */
export async function embedQuery(query: string): Promise<number[]> {
  const extractor = await getExtractor();
  const out = await extractor(BGE_QUERY_PREFIX + query, {
    pooling: "cls",
    normalize: true,
  });
  return Array.from(out.data as Float32Array);
}
