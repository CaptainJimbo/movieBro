/**
 * Dense retrieval leg: query vector -> Cloudflare Worker -> Pinecone.
 *
 * The Worker is a pure proxy (holds the API key server-side); on ANY
 * failure the caller falls back to BM25-only mode per the free-tier
 * resilience spec — the search bar must never die.
 */

import { LEG_TOP_K, WORKER_URL } from "./config";
import type { LegHit } from "./types";

/**
 * Query the dense leg via the Worker.
 *
 * @param vector - normalized 384-dim query embedding.
 * @param topK - matches to request (defaults to the leg standard).
 * @returns Hits sorted by cosine score desc.
 * @throws Error when the Worker is unreachable or returns non-200 —
 *   callers catch and degrade to lexical-only.
 */
export async function searchDense(
  vector: number[],
  topK: number = LEG_TOP_K,
): Promise<LegHit[]> {
  const r = await fetch(WORKER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vector, topK }),
  });
  if (!r.ok) throw new Error(`dense leg ${r.status}`);
  const { matches } = (await r.json()) as { matches: { id: string; score: number }[] };
  return matches.map((m) => ({ childId: Number(m.id), score: m.score }));
}
