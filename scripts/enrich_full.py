#!/usr/bin/env python3
"""Enrich the FULL MovieLens catalog via TMDB -> data/build/enriched.jsonl.

Concurrent (10 workers), resumable: already-enriched movieIds are skipped on
re-run, so failures can be retried by simply running again. Failures are
logged with a reason and reported at the end.
"""

import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "movielens"
OUT = ROOT / "data" / "build"
OUT_PATH = OUT / "enriched.jsonl"
WORKERS = 10

write_lock = threading.Lock()


def load_catalog() -> list[dict]:
    """Join movies.csv + links.csv + rating counts into base movie records.

    Returns:
        One dict per MovieLens movie: movieId (int), title (with year
        suffix, MovieLens style), genres (list), tmdbId (str or None when
        links.csv has no mapping), numRatings (popularity signal for the
        onboarding wall and fallback ranking).
    """
    with open(ML / "links.csv", newline="") as f:
        tmdb = {r["movieId"]: r["tmdbId"] for r in csv.DictReader(f)}
    with open(ML / "ratings.csv", newline="") as f:
        counts: dict[str, int] = {}
        for r in csv.DictReader(f):
            counts[r["movieId"]] = counts.get(r["movieId"], 0) + 1
    with open(ML / "movies.csv", newline="") as f:
        return [
            {"movieId": int(r["movieId"]), "title": r["title"],
             "genres": r["genres"].split("|"),
             "tmdbId": tmdb.get(r["movieId"]) or None,
             "numRatings": counts.get(r["movieId"], 0)}
            for r in csv.DictReader(f)
        ]


def enrich_one(movie: dict, api_key: str, session: requests.Session) -> tuple[dict | None, str]:
    """Fetch one movie's TMDB enrichment (details+keywords+credits, 1 call).

    Args:
        movie: base record from load_catalog(); needs "tmdbId".
        api_key: TMDB v3 API key.
        session: shared requests.Session (connection pooling across threads).

    Returns:
        (record, "") on success — the base record extended with posterPath,
        overview, tagline, keywords (<=15), cast (top 5), directors, year.
        (None, reason) on failure — missing tmdbId, HTTP error, or network
        error; the reason string lands in enrich_failures.json.
    """
    if not movie["tmdbId"]:
        return None, "no tmdbId in links.csv"
    try:
        r = session.get(
            f"https://api.themoviedb.org/3/movie/{movie['tmdbId']}",
            params={"api_key": api_key, "append_to_response": "keywords,credits"},
            timeout=20,
        )
    except requests.RequestException as e:
        return None, f"request error: {e}"
    if r.status_code == 404:
        return None, "tmdb 404"
    if r.status_code != 200:
        return None, f"tmdb {r.status_code}"
    d = r.json()
    return {
        **movie,
        "posterPath": d.get("poster_path"),
        "overview": d.get("overview") or "",
        "tagline": d.get("tagline") or "",
        "keywords": [k["name"] for k in d.get("keywords", {}).get("keywords", [])[:15]],
        "cast": [c["name"] for c in d.get("credits", {}).get("cast", [])[:5]],
        "directors": [c["name"] for c in d.get("credits", {}).get("crew", [])
                      if c.get("job") == "Director"],
        "year": (d.get("release_date") or "")[:4],
    }, ""


def main() -> int:
    """Enrich every not-yet-enriched movie concurrently; append to JSONL.

    Resumability: movieIds already present in enriched.jsonl are skipped,
    so re-running retries only past failures. WORKERS threads share one
    session; each success is appended under write_lock (JSONL keeps
    partial runs valid). Failures are summarized on stdout and dumped to
    enrich_failures.json for inspection.

    Returns:
        Process exit code (0 on success, 1 if TMDB_API_KEY is missing).
    """
    api_key = dotenv_values(ROOT / ".env").get("TMDB_API_KEY")
    if not api_key:
        print("TMDB_API_KEY missing", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            done = {json.loads(line)["movieId"] for line in f if line.strip()}

    todo = [m for m in load_catalog() if m["movieId"] not in done]
    print(f"already enriched: {len(done)} | to do: {len(todo)}", flush=True)

    failures: list[tuple[int, str, str]] = []
    session = requests.Session()
    n_ok = 0
    with open(OUT_PATH, "a") as out, ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(enrich_one, m, api_key, session): m for m in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            movie = futures[fut]
            rec, err = fut.result()
            if rec:
                with write_lock:
                    out.write(json.dumps(rec) + "\n")
                n_ok += 1
            else:
                failures.append((movie["movieId"], movie["title"], err))
            if k % 500 == 0:
                out.flush()
                print(f"  {k}/{len(todo)} ({n_ok} ok, {len(failures)} failed)", flush=True)

    print(f"\nenriched this run: {n_ok} | failures: {len(failures)}")
    for mid, title, err in failures[:20]:
        print(f"  FAIL {mid} {title!r}: {err}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more")
    (OUT / "enrich_failures.json").write_text(json.dumps(failures, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
