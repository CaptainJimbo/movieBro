#!/usr/bin/env python3
"""Step-1 data gate: enrich the 100 most-rated MovieLens movies via TMDB.

Proves the whole chain cheaply before the full ~9k run:
  links.csv join resolves -> TMDB API works -> posters render -> data quality OK.

Outputs (both regenerable, gitignored under data/raw/):
  data/raw/sample100.json  - enriched records
  data/raw/sample_grid.html - static poster grid, open locally as the gate proof
"""

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "movielens"
OUT = ROOT / "data" / "raw"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"
SAMPLE_SIZE = 100


def load_top_movies(n: int) -> list[dict]:
    """The n most-rated movies, joined with their TMDB ids."""
    with open(ML / "ratings.csv", newline="") as f:
        counts = Counter(row["movieId"] for row in csv.DictReader(f))

    with open(ML / "links.csv", newline="") as f:
        tmdb_by_movie = {row["movieId"]: row["tmdbId"] for row in csv.DictReader(f)}

    with open(ML / "movies.csv", newline="") as f:
        movies = {row["movieId"]: row for row in csv.DictReader(f)}

    top = []
    for movie_id, n_ratings in counts.most_common(n):
        m = movies[movie_id]
        top.append(
            {
                "movieId": int(movie_id),
                "title": m["title"],
                "genres": m["genres"].split("|"),
                "tmdbId": tmdb_by_movie.get(movie_id) or None,
                "numRatings": n_ratings,
            }
        )
    return top


def enrich(movie: dict, api_key: str, session: requests.Session) -> dict | None:
    """One TMDB call per movie: details + keywords + top cast."""
    if not movie["tmdbId"]:
        return None
    r = session.get(
        f"https://api.themoviedb.org/3/movie/{movie['tmdbId']}",
        params={"api_key": api_key, "append_to_response": "keywords,credits"},
        timeout=15,
    )
    if r.status_code != 200:
        return None
    d = r.json()
    directors = [c["name"] for c in d["credits"]["crew"] if c["job"] == "Director"]
    return {
        **movie,
        "posterPath": d.get("poster_path"),
        "overview": d.get("overview") or "",
        "tagline": d.get("tagline") or "",
        "keywords": [k["name"] for k in d["keywords"]["keywords"][:15]],
        "cast": [c["name"] for c in d["credits"]["cast"][:5]],
        "directors": directors,
        "tmdbTitle": d.get("title"),
        "year": (d.get("release_date") or "")[:4],
    }


def render_grid(records: list[dict]) -> str:
    cards = "\n".join(
        f'<figure><img src="{POSTER_BASE}{r["posterPath"]}" loading="lazy" '
        f'alt="{r["title"]}"><figcaption>{r["title"]}</figcaption></figure>'
        for r in records
        if r.get("posterPath")
    )
    return f"""<!doctype html>
<meta charset="utf-8"><title>movieBro — step-1 data gate</title>
<style>
  body {{ background:#111; color:#eee; font:14px system-ui; margin:2rem }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:12px }}
  figure {{ margin:0 }} img {{ width:100%; border-radius:8px; display:block }}
  figcaption {{ padding:4px 2px; font-size:12px; color:#aaa }}
</style>
<h1>movieBro — 100 most-rated, enriched via TMDB</h1>
<p>Posters hotlinked from TMDB CDN. This page is the step-1 gate artifact.</p>
<div class="grid">{cards}</div>
"""


def main() -> int:
    env = dotenv_values(ROOT / ".env")
    api_key = env.get("TMDB_API_KEY")
    if not api_key:
        print("TMDB_API_KEY missing from .env", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    top = load_top_movies(SAMPLE_SIZE)

    session = requests.Session()
    enriched, failures = [], []
    for i, movie in enumerate(top, 1):
        rec = enrich(movie, api_key, session)
        if rec:
            enriched.append(rec)
        else:
            failures.append(movie["title"])
        if i % 20 == 0:
            print(f"  {i}/{len(top)} enriched...")
        time.sleep(0.05)

    (OUT / "sample100.json").write_text(json.dumps(enriched, indent=2))
    (OUT / "sample_grid.html").write_text(render_grid(enriched))

    no_poster = [r["title"] for r in enriched if not r.get("posterPath")]
    no_overview = [r["title"] for r in enriched if not r["overview"]]
    print(f"\nenriched : {len(enriched)}/{len(top)}")
    print(f"failures : {failures or 'none'}")
    print(f"no poster: {no_poster or 'none'}")
    print(f"no overview: {no_overview or 'none'}")
    kw = sum(len(r["keywords"]) for r in enriched) / max(len(enriched), 1)
    print(f"avg keywords/movie: {kw:.1f}")
    print(f"\nwrote {OUT / 'sample100.json'}")
    print(f"wrote {OUT / 'sample_grid.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
