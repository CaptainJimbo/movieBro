# movieBro — Project Spec

**Recommender + hybrid-search demo over movies** — item-item CF with in-browser
fold-in, parent-child dense+BM25→RRF retrieval, in-browser cross-encoder
rerank. Private while under construction; public at launch (Pages needs it).

Read `CLAUDE.local.md` first if it exists (local-only context; gitignored).

## What this project demonstrates (the point)

The **matching & recommendation** skill axis, publicly: collaborative
filtering, hybrid retrieval with parent-child chunking, rank fusion,
reranking, and honest offline evaluation — end to end, fully client-side.
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
- **Keys:** TMDB key is used at BUILD TIME only (worker/local script) — never
  ships to the browser. Lives in `.env` locally / Actions secret if the build
  runs in CI.

## Architecture

### Build time (Python, offline — local or Actions)
1. **Ingest:** MovieLens CSVs + TMDB enrichment → SQLite as the working store
   (SQLite is a BUILD artifact, not a server — nothing queries it at runtime).
2. **CF training:** item-item collaborative filtering — mean-centered cosine
   over the user-item matrix (classic Sarwar et al. 2001), shrinkage toward 0
   for low-support pairs (min ~5 co-raters), keep **top-50 neighbors per
   movie** → `neighbors.json` (~9k × 50 ids+sims, ~2–3 MB). Optional v2:
   implicit ALS embeddings if cosine disappoints in eval.
3. **Search index:** parent = movie; children = overview sentences, tag
   clusters, keywords, tagline. Embed children with **bge-small-en-v1.5**
   (384-dim; same model family + query-prefix discipline as limenarchisAI) →
   fp16 vectors + BM25 term stats (k1=1.2, b=0.75, Lucene negative-IDF floor)
   → `search-index.json` (~10-15 MB budget for ml-latest-small; measure).
4. **Catalog:** `movies.json` — title, year, genres, poster path, TMDB id.

### Runtime (browser only — GitHub Pages, zero backend)
- **Onboarding:** poster wall (popular-but-diverse seed set, stratified by
  genre) → user rates ≥10 → ratings in localStorage. NOTHING leaves the device.
- **Recommendations:** fold-in — for each candidate movie,
  score = Σ(sim × (user_rating − user_mean)) over rated neighbors, normalized
  by Σ|sim|; exclude rated movies. Top-3 user genres from rating-weighted
  genre counts → **one CF pick per genre** on the dashboard ("your next
  watch"), with a "because you rated X, Y" explanation (the neighbors that
  drove the score — provenance for recsys).
- **Search:** query → transformers.js embedding + JS BM25 over children →
  **RRF k=60** → **best-hit grouping to parent (max, never sum)** → top ~20
  parents → **cross-encoder rerank in-browser** (ms-marco-MiniLM ONNX,
  query × [title + overview]) → **3×3 poster grid**. Hover: why it matched —
  matching child snippet + D/B leg badges (provenance UI, the money-shot).
- **Blend mode (the signature feature):** search results get a subtle
  personal boost from the CF score (weighted blend, default small, slider in
  a debug panel) — search that knows your taste. Label it visibly.

## Evaluation (EVALUATION.md, house style — do not skip)

- **CF:** temporal split per user (last 20% of each user's ratings held out).
  Metrics: hit-rate@10, NDCG@10 vs (a) popularity baseline, (b) genre-matched
  popularity. If CF doesn't beat popularity, SAY SO and investigate shrinkage.
- **Search:** golden set of ~50 queries (hand-written: exact-title, actor-ish,
  vibe queries, constraint queries) with graded relevant movies. Report
  recall@9 / MRR per leg (dense-only, bm25-only, fused, fused+rerank) — show
  what each stage buys or state that it doesn't.
- **Known limitation to state plainly:** ml-latest-small is 610 users — CF
  will be noisy for niche tastes; the fix (ml-25m) is the documented upgrade
  path.

## Build spine (each step a working artifact)

1. **Data gate:** download MovieLens, get TMDB key, enrich 100 movies,
   render a static poster grid locally. Proves data + posters + licenses.
2. **CF offline:** train item-item, run the eval, beat popularity (or learn
   why not). EVALUATION.md starts here, before any UI.
3. **Search index + retrieval in browser:** children, embeddings, BM25, RRF,
   grouping — search bar returns the 3×3 grid. Provenance hover.
4. **Rerank:** cross-encoder on top-20; eval table gets the +rerank column.
5. **Onboarding + dashboard:** rating flow, localStorage, fold-in recs,
   top-3-genre picks with "because you rated" explanations.
6. **Blend mode + polish** (frontend-design pass) + final EVALUATION.md.
7. **Publish:** repo public, Pages live, portfolio card, demo GIF.

## Performance budgets

- Shipped payload: catalog + neighbors + search index ≤ ~20 MB total
  (lazy-load index on first search; catalog first paint < 1s).
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
  Greek subtitles/UI, Letterboxd import.
- This is a bounded demo: 7 steps, then back to the fleet.

## Related repos

- `limenarchisAI` — retrieval-stack sibling (bge-small discipline, BM25/RRF
  constants, provenance-UI ideas — keep implementations consistent).
- `o-ilios` — EVALUATION.md house style, README voice.
- `polish-my-profile` — master queue (`PORTFOLIO_PLAN.md`).
- `CaptainJimbo.github.io` — portfolio site; gets a card at launch.
