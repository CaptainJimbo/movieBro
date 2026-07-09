# movieBro — Project Spec

**Recommender + hybrid-search demo over movies** — item-item CF with in-browser
fold-in, parent-child dense+BM25→RRF retrieval, in-browser cross-encoder
rerank. Private while under construction; public at launch (Pages needs it).

Read `CLAUDE.local.md` first if it exists (local-only context; gitignored).

## What this project demonstrates (the point)

The **matching & recommendation** skill axis, publicly: collaborative
filtering, hybrid retrieval with parent-child chunking, rank fusion,
reranking, and honest offline evaluation — end to end, client-side except a
thin Worker proxy for the managed dense-vector lookup (Pinecone).
Sibling of `limenarchisAI` (RAG chat) — same retrieval family, different
function: that one answers questions with citations; this one **matches and
recommends**.

## Data (all free; verify at step 1)

- **MovieLens `ml-latest-small`** (9k movies / 100k ratings / 3.6k tags) for
  v1 — the demo-sized standard. Upgrade path: `ml-25m` + **tag genome**
  (1,128 scored tags per movie — search-children gold) if build times stay sane.
  License: research/non-commercial, cite Harper & Konstan 2015 (in README).
- **TMDB API** (free key) via MovieLens `links.csv` → posters, overview,
  keywords, cast toplines. Attribution required (in README). Poster URLs are
  hotlinked from TMDB's CDN (allowed) — do NOT commit poster images.
- **Keys (two, different lifetimes, neither ever in browser JS):**
  - **TMDB key** — BUILD TIME only (local build script) → `.env` locally /
    Actions secret if the build runs in CI.
  - **`PINECONE_API_KEY`** — RUNTIME, but server-side only: a **Cloudflare
    Worker secret** the proxy reads. The browser calls the Worker, never
    Pinecone directly, so this key never reaches client JS.

## Architecture

### Build time (Python, offline — local or Actions)
1. **Ingest:** MovieLens CSVs + TMDB enrichment → SQLite as the working store
   (SQLite is a BUILD artifact, not a server — nothing queries it at runtime).
2. **CF training:** item-item collaborative filtering — adjusted cosine
   (user-mean-centered) over the user-item matrix (classic Sarwar et al.
   2001), similarity shrinkage **n/(n+β) with β=400** (eval: broad plateau
   β≈100–800; single digits are folk-noise with 610 users — EVALUATION.md),
   keep **top-50 neighbors per movie** → `neighbors.json` (measured:
   **7.25 MB** minified; within total budget). Optional v2: implicit ALS
   embeddings if cosine disappoints in eval. [Step 2 DONE: CF beats
   popularity, HR@10 0.42 vs 0.32.]
3. **Search index:** parent = movie; children = overview sentences, tag
   clusters, keywords, tagline. Embed children with **bge-small-en-v1.5**
   (384-dim; same model family + query-prefix discipline as limenarchisAI).
   Dense vectors are **upserted to Pinecone** (build script, key local) —
   they do NOT ship to the browser. What ships is `search-index.json`: BM25
   term stats (k1=1.2, b=0.75, Lucene negative-IDF floor) + child text +
   child→parent mapping (~few MB now that vectors live in Pinecone; measure).
4. **Catalog:** `movies.json` — title, year, genres, poster path, TMDB id.

### Runtime (browser + thin Cloudflare Worker proxy for dense lookup)
Everything runs client-side on GitHub Pages EXCEPT the dense ANN lookup,
which goes through a thin Cloudflare Worker that holds `PINECONE_API_KEY`
server-side (browser sends only the query vector; never a key). All ranking
— BM25, RRF, grouping, rerank — stays in the browser.

**Free-tier resilience (non-negotiable):** Pinecone Starter *pauses idle
indexes after ~3 weeks of inactivity* — fatal for a demo that sits idle
between recruiter visits. Two mitigations, both free: (a) **keep-alive** —
a Cloudflare **Cron Trigger** (free plan includes them) pings the index via
the Worker weekly; (b) **graceful degradation** — if the dense-leg fetch
fails for any reason, search falls back to **BM25-only → rerank**, visibly
badged ("lexical-only mode"), so the search bar never dies. Verify the
pause policy against current Pinecone docs when the Worker stands up (step 3).
- **Onboarding:** poster wall (popular-but-diverse seed set, stratified by
  genre). Three actions per poster: **❤️ like (+1)**, **🥔 nope (−1)**,
  **⏭️ haven't-seen (no signal, swaps in a fresh poster)**. Gate on **≥10
  real reactions** (❤️/🥔; skips don't count), **open ceiling** — "keep going,
  more is better" (more ratings average over more neighbors, damping — not
  eliminating — the 610-user sparsity noise). Grid
  layout, so not-reacting is already an implicit skip; the ⏭️ button just
  swaps for a recognizable title to keep onboarding productive. Ratings in
  localStorage. NOTHING leaves the device.
- **Recommendations:** fold-in — for each candidate movie,
  score = Σ(sim × (user_rating − user_mean)) over rated neighbors,
  **UNNORMALIZED** (do NOT divide by Σ|sim| — the eval showed normalization
  collapses top-N ranking, HR@10 0.42→0.01: it turns the score into a rating
  prediction and low-evidence candidates flood the top; EVALUATION.md
  finding 2); exclude rated movies. Top-3 user genres from rating-weighted
  genre counts → **one CF pick per genre** on the dashboard ("your next
  watch"), with a "because you rated X, Y" explanation (the neighbors that
  drove the score — provenance for recsys). **Every dashboard card is ratable
  (❤️/🥔)** → writes localStorage → fold-in re-runs live (O(rated × neighbors),
  trivial); don't reshuffle violently mid-session (update on next visit or
  behind a subtle "refresh picks"). Note: item-item similarities
  (`neighbors.json`) are FIXED at build time — more ratings sharpen the
  *fold-in / personalization*, they do NOT retrain the model.
  **Binary-scale caveat (❤️/🥔 = ±1):** a user with ONLY hearts (or only
  potatoes) makes (rating − user_mean) = 0 for every movie → fold-in
  collapses to all-zeros. Fix: **damp the user mean toward 0**
  (ū = Σr/(n+β), β≈5) — principled here because the ±1 scale is already
  symmetric about 0; the damping only removes generosity bias gradually as
  evidence accumulates. (The offline eval uses MovieLens 5-star data with
  true means — this caveat is for live users only.)
- **"More like this" strip (item-item, not personalized):** under a movie,
  a labeled row of its CF neighbors from `neighbors.json` ("viewers also
  liked") — nearly free once step 2 exists, so it lands in **v1 (step 5)**.
  The content-based *same-cast/director* variant ("more with Russell Crowe")
  is **v2 parking lot** (needs cast metadata plumbing). Keep it OUT of the
  search relevance grid — it answers "similar to this movie," not the query.
- **Search:** query → transformers.js embedding → **dense leg via Worker→
  Pinecone** (returns child ids+scores) ‖ **JS BM25 over children** (client)
  → **RRF k=60** → **best-hit grouping to parent (max, never sum)** → top ~20
  parents → **cross-encoder rerank in-browser** (ms-marco-MiniLM ONNX,
  query × [title + overview]) → **3×3 poster grid**. Hover: why it matched —
  matching child snippet + D/B leg badges (provenance UI, the money-shot).
- **Blend mode (the signature feature):** the ONE bridge between the
  retrieval side (content-based) and the recommender side (item-item CF).
  Applied **after rerank**, re-ordering ONLY the ~20 search candidates —
  never injects CF-only movies (the query still decides relevance; CF only
  nudges order). Formula is a **convex combination** of normalized scores:
  `final = (1−α)·sigmoid(rerank) + α·sigmoid(cf/τ)`. Sigmoid (not min-max) →
  fixed scale, and **cf=0 → 0.5 = neutral**; cf is the UNNORMALIZED fold-in
  score, so its spread grows with ratings count — τ (temperature) calibrated
  at step 6 so typical scores land in sigmoid's active range. **α = slider** (debug panel),
  default small (~0.15); **CF-silent candidate (no rated neighbors) → 0.5,
  never penalized**; **pre-onboarding → α=0** (pure search). Provenance hover
  shows the nudge ("▲ because you ❤️ Heat, Sicario"). Label it visibly.
  - *Justification:* convex combination is the robust, single-parameter,
    normalization-agnostic fusion choice when you can't train a full ranker
    (Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid
    Retrieval," TOIS 2023 — CC beats RRF, RRF sensitive to its k).
  - *Honest upgrade path (state in EVALUATION.md):* the textbook-best blend
    is **learning-to-rank with the CF score as a feature** (LambdaMART / a
    learned re-ranker); deferred because it needs click/relevance labels this
    demo doesn't collect (the Worker is a stateless proxy that logs nothing;
    ratings/taste stay on-device).
  - *Eval caveat:* blend perturbs query-relevance, so it can only make the
    search golden-set numbers worse — do NOT eval blend against them. The
    debug-panel α slider + side-by-side "search-only vs blended" IS the
    demonstration; personalized-search quality has no offline ground truth
    here (would need user studies) — say so plainly.

### Frontend stack — vanilla TypeScript + Vite (no framework)
Pages serves static only; the UI surface is small (poster wall, dashboard,
search + 3×3 grid, debug panel) while the real complexity is the retrieval/
CF math, which should read as pure TS — not framework plumbing. Vite gives
TS, dev server, tree-shaken build, and `base: '/movieBro/'` for Pages; deploy
via the standard Actions workflow (build → `dist/` → Pages). If UI state ever
genuinely outgrows hand-rendering, Preact (~3 KB) is the contained fallback —
a refactor, not a rewrite. React/SSR rejected: no server, no state complexity
to justify the payload.

## Evaluation (EVALUATION.md, house style — do not skip)

- **CF:** temporal split per user (last 20% of each user's ratings held out).
  Metrics: hit-rate@10, NDCG@10 vs (a) popularity baseline, (b) genre-matched
  popularity. If CF doesn't beat popularity, SAY SO and investigate shrinkage.
- **Search:** golden set of ~50 queries (hand-written: exact-title, actor-ish,
  vibe queries, constraint queries) with graded relevant movies. Report
  recall@9 / MRR per leg (dense-only, bm25-only, fused, fused+rerank) — show
  what each stage buys or state that it doesn't. *Optional v2 experiment:*
  add a convex-combination fusion row alongside RRF k=60 (Bruch et al. 2023
  suggest CC ≥ RRF) — cheap, and shows current-literature awareness.
- **Known limitation to state plainly:** ml-latest-small is 610 users — CF
  will be noisy for niche tastes; the fix (ml-25m) is the documented upgrade
  path.

## Build spine (each step a working artifact)

1. **Data gate:** download MovieLens, get TMDB key, enrich 100 movies,
   render a static poster grid locally. Proves data + posters + licenses.
2. **CF offline:** train item-item, run the eval, beat popularity (or learn
   why not). EVALUATION.md starts here, before any UI.
3. **Search index + retrieval:** children, embeddings (build-time upsert to
   Pinecone), BM25 stats shipped; dense leg via Worker→Pinecone, BM25/RRF/
   grouping in browser — search bar returns the 3×3 grid. Provenance hover.
   (Cloudflare Worker proxy holding `PINECONE_API_KEY` stands up here.)
4. **Rerank:** cross-encoder on top-20; eval table gets the +rerank column.
5. **Onboarding + dashboard:** poster wall with ❤️/🥔/⏭️ (≥10, open ceiling),
   localStorage, live fold-in recs, top-3-genre picks with "because you
   rated" explanations, ❤️/🥔 on every dashboard card, "more like this"
   CF-neighbor strip.
6. **Blend mode + polish** (frontend-design pass) + final EVALUATION.md.
7. **Publish:** repo public, Pages live, portfolio card, demo GIF.

## Performance budgets

- Shipped payload: catalog + neighbors + search index (BM25 stats + child
  text; dense vectors live in Pinecone, not shipped) — well under the old
  ~20 MB budget now; lazy-load index on first search; catalog first paint < 1s.
- Embedding model (~25 MB) + cross-encoder (~23 MB) lazy-loaded with progress
  UI; cache via service worker if trivial, else browser cache is fine.
- All scoring loops are O(rated × neighbors) or O(children) — trivial in JS.

## Plugins for this repo (recommend at session start)

```
claude plugin enable frontend-design   # poster-wall + dashboard polish
claude plugin install pyright-lsp@claude-plugins-official
```

Already at user scope: playwright, chrome-devtools (screenshot-driven UI
iteration), context7 (transformers.js/onnxruntime-web docs), firecrawl.

## Working conventions (house rules)

- Batch edits, one commit. Ask before slow deploys.
- Step-1 data gate before building on top (the FLOGA lesson).
- Eval before UI polish (step 2 precedes step 5 for a reason).
- New ideas → v2 parking lot: ml-25m + tag genome, implicit ALS, "surprise
  me" diversity knob (MMR), friend-blend (two people rate → joint picks),
  Greek subtitles/UI, Letterboxd import, **content-based "more like this"
  (same cast/director, from TMDB metadata)**, **learning-to-rank blend (CF
  score as a feature, if a backend ever logs clicks)**, **convex-combination
  fusion row in the search eval**.
- This is a bounded demo: 7 steps, then back to the fleet.

## Related repos

- `limenarchisAI` — retrieval-stack sibling (bge-small discipline, BM25/RRF
  constants, provenance-UI ideas — keep implementations consistent).
- `o-ilios` — EVALUATION.md house style, README voice.
- `CaptainJimbo.github.io` — portfolio site; gets a card at launch.
