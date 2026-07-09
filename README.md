# movieBro 🎬🍿

**Your movie bro.** Rate a few movies, and he learns your taste — then serves
your next watch, one pick per favorite genre, plus a search bar that
understands *"mind-bending sci-fi like Inception but sadder."*

> Not a critic. A bro. He doesn't judge your taste — he feeds it.

## What it does

1. **Onboarding** — rate ~10 movies (posters, star taps). That's the only input.
2. **"Your next watch" dashboard** — your top 3 genres inferred from your
   ratings, one **collaborative-filtering** pick per genre: movies loved by
   people who rated like you, not just popular stuff.
3. **The search bar** — free-text, vibe-friendly. Hybrid retrieval (semantic +
   keyword) over plot, tags, and keywords → a **3×3 grid of matched movies**,
   reranked by a cross-encoder. Hover a poster to see *why* it matched.
4. **Honest evaluation** — held-out ratings, hit-rate@10 / NDCG vs a
   popularity baseline, published in EVALUATION.md. If the fancy stuff doesn't
   beat "just recommend popular movies," the table will say so.

## How it works (all in your browser — no backend, no accounts, no tracking)

- **Recommendations:** item-item collaborative filtering trained offline on
  **MovieLens** ratings; each movie ships with its top-neighbor list, and your
  ratings are folded in client-side (score = Σ similarity × your stars).
  Your ratings never leave your device.
- **Search:** movies are *parents*; plot sentences, crowd tags and keywords
  are *children*. Query → dense vectors (transformers.js) **+ BM25** (in JS)
  → **Reciprocal Rank Fusion** → best-hit grouping to the parent movie →
  **cross-encoder rerank** of the finalists (also in-browser, ONNX).
- **Serving:** everything is precomputed into static files at build time and
  hosted on GitHub Pages. Total recurring cost: €0.

## Status

🚧 Spec'd, not yet built — see `CLAUDE.md` for architecture and build plan.

## Data & credits

- **[MovieLens](https://grouplens.org/datasets/movielens/)** (GroupLens
  Research) — ratings, tags, tag genome. Used under their non-commercial
  research license, with citation: F. M. Harper & J. A. Konstan, *The
  MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015.
- **[TMDB](https://www.themoviedb.org/)** — posters, overviews, keywords via
  their free API. *This product uses the TMDB API but is not endorsed or
  certified by TMDB.*

## License

MIT © 2026 Dimitris Kogias (code; datasets keep their own licenses)

---

*Built by [Dimitris Kogias](https://captainjimbo.github.io) — physicist &
AI/ML systems engineer. Siblings:
[Ο Ήλιος](https://github.com/CaptainJimbo/o-ilios) ·
[ArcheoLogic](https://github.com/CaptainJimbo/archeologic) ·
[pyroPythia](https://github.com/CaptainJimbo/pyroPythia) ·
[LimenarchisAI](https://github.com/CaptainJimbo/limenarchisAI).*
