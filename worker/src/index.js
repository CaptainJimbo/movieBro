/**
 * movieBro dense-search proxy (Cloudflare Worker).
 *
 * A pure pass-through in front of Pinecone: the browser embeds the query
 * with bge-small locally and POSTs just the 384-dim vector here; this
 * Worker attaches PINECONE_API_KEY (Worker secret) and forwards to the
 * index's /query endpoint, returning [{id, score}] matches. No app logic,
 * no storage, no user data — ratings never leave the device.
 *
 * Also handles the weekly cron keep-alive so Pinecone's free tier never
 * pauses the index for inactivity.
 */

const DIM = 384;
const MAX_TOP_K = 100;
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

/**
 * Build a JSON Response with CORS headers attached.
 *
 * @param {unknown} body - JSON-serializable payload.
 * @param {number} [status=200] - HTTP status code.
 * @returns {Response} the finished response.
 */
function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...CORS },
  });
}

/**
 * Query Pinecone's data-plane /query endpoint.
 *
 * @param {object} env - Worker env (PINECONE_HOST var, PINECONE_API_KEY secret).
 * @param {number[]} vector - normalized 384-dim query embedding.
 * @param {number} topK - number of matches to return.
 * @returns {Promise<{matches: {id: string, score: number}[]}>} raw Pinecone body.
 */
async function pineconeQuery(env, vector, topK) {
  const r = await fetch(`${env.PINECONE_HOST}/query`, {
    method: "POST",
    headers: {
      "Api-Key": env.PINECONE_API_KEY,
      "Content-Type": "application/json",
      "X-Pinecone-API-Version": "2025-01",
    },
    body: JSON.stringify({ vector, topK, includeMetadata: false }),
  });
  if (!r.ok) throw new Error(`pinecone ${r.status}: ${await r.text()}`);
  return r.json();
}

export default {
  /**
   * HTTP entrypoint: OPTIONS preflight + POST {vector, topK?} -> matches.
   *
   * Validates the vector strictly (length, finite numbers) so the proxy
   * can't be used to relay arbitrary payloads; everything else is a 4xx.
   *
   * @param {Request} request - incoming request.
   * @param {object} env - Worker env bindings.
   * @returns {Promise<Response>} JSON matches or an error body.
   */
  async fetch(request, env) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
    if (request.method !== "POST") return json({ error: "POST only" }, 405);

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "invalid JSON" }, 400);
    }
    const { vector, topK = 60 } = body ?? {};
    if (!Array.isArray(vector) || vector.length !== DIM ||
        !vector.every((x) => typeof x === "number" && Number.isFinite(x))) {
      return json({ error: `vector must be ${DIM} finite numbers` }, 400);
    }
    const k = Math.min(Math.max(1, topK | 0), MAX_TOP_K);

    try {
      const out = await pineconeQuery(env, vector, k);
      return json({
        matches: (out.matches ?? []).map((m) => ({ id: m.id, score: m.score })),
      });
    } catch (e) {
      return json({ error: String(e) }, 502);
    }
  },

  /**
   * Weekly cron keep-alive: one throwaway query counts as index activity,
   * preventing the free-tier inactivity pause. Failures are logged only —
   * next week's tick retries.
   *
   * @param {ScheduledController} _controller - cron metadata (unused).
   * @param {object} env - Worker env bindings.
   */
  async scheduled(_controller, env) {
    try {
      await pineconeQuery(env, new Array(DIM).fill(0.001), 1);
      console.log("keep-alive ok");
    } catch (e) {
      console.error("keep-alive failed:", e);
    }
  },
};
