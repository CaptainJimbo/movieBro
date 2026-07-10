#!/usr/bin/env python3
"""Build the shipped search artifacts from the enriched catalog.

Outputs (data/build/, regenerable):
  movies.json       - catalog: every MovieLens movie (poster/year when
                      enriched), the app's first-paint payload.
  search-index.json - flat child list [[movieId, type, text], ...]; the
                      browser builds BM25 over it at load time. A child's
                      array position IS its id (and its Pinecone vector id).

Children per movie (parent = movie, per spec): overview sentences, tagline,
keywords, MovieLens community tags, cast+director topline.
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "movielens"
BUILD = ROOT / "data" / "build"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
MIN_SENT_CHARS = 20
MAX_OVERVIEW_SENTS = 5


def clean_title(ml_title: str) -> tuple[str, str]:
    """Split a MovieLens title into display title and year.

    MovieLens format is "Title (Year)" with an optional ", The/A/An"
    suffix moved to the end ("Matrix, The (1999)"). Returns the natural
    reading order ("The Matrix") and the year string ("" if absent).

    Args:
        ml_title: raw movies.csv title field.

    Returns:
        (title, year) — e.g. ("The Matrix", "1999").
    """
    m = re.match(r"^(.*?)(?:\s*\((\d{4})\))?\s*$", ml_title)
    title, year = (m.group(1) or ml_title).strip(), m.group(2) or ""
    art = re.match(r"^(.*),\s*(The|A|An|Les|Le|La|Der|Die|Das|El|Los|Il)$", title)
    if art:
        title = f"{art.group(2)} {art.group(1)}"
    return title, year


def load_enriched() -> dict[int, dict]:
    """Load enriched.jsonl into a movieId-keyed dict.

    Returns:
        {movieId: enriched record} for every successfully enriched movie.
    """
    out = {}
    with open(BUILD / "enriched.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[r["movieId"]] = r
    return out


def load_ml_tags() -> dict[int, list[str]]:
    """Aggregate MovieLens community tags per movie (deduped, lowercased).

    Returns:
        {movieId: [unique tag strings]} — only movies that have tags.
    """
    tags = defaultdict(dict)  # movieId -> {lowered: original} keeps first casing
    with open(ML / "tags.csv", newline="") as f:
        for r in csv.DictReader(f):
            t = r["tag"].strip()
            if t:
                tags[int(r["movieId"])].setdefault(t.lower(), t)
    return {m: list(d.values()) for m, d in tags.items()}


def children_for(title: str, year: str, movie: dict,
                 ml_tags: list[str]) -> list[tuple[str, str]]:
    """Extract search children (type, text) for one movie.

    Child types: "title" (always present — the lexical anchor for
    exact-title queries, which no other child covers), "overview" (per
    sentence, capped), "tagline", "keywords" (one joined child), "tags"
    (community tags, joined), "cast" (actors + director topline — covers
    actor-ish queries).

    Args:
        title: display title from clean_title() ("The Matrix").
        year: release year string ("" if unknown).
        movie: enriched record ({} for unenriched movies).
        ml_tags: this movie's MovieLens tag list ([] if none).

    Returns:
        List of (type, text) pairs; at minimum the title child.
    """
    kids: list[tuple[str, str]] = [
        ("title", f"{title} ({year})" if year else title)
    ]
    for sent in SENT_SPLIT.split(movie.get("overview") or "")[:MAX_OVERVIEW_SENTS]:
        if len(sent.strip()) >= MIN_SENT_CHARS:
            kids.append(("overview", sent.strip()))
    if movie.get("tagline"):
        kids.append(("tagline", movie["tagline"].strip()))
    if movie.get("keywords"):
        kids.append(("keywords", ", ".join(movie["keywords"])))
    if ml_tags:
        kids.append(("tags", ", ".join(ml_tags[:25])))
    cast, directors = movie.get("cast") or [], movie.get("directors") or []
    if cast or directors:
        parts = []
        if cast:
            parts.append("starring " + ", ".join(cast))
        if directors:
            parts.append("directed by " + ", ".join(directors))
        kids.append(("cast", "; ".join(parts)))
    return kids


def main() -> int:
    """Assemble movies.json + search-index.json from all build inputs.

    Every MovieLens movie gets a catalog entry (unenriched ones lack
    poster/overview fields but stay recommendable by CF); children are
    emitted only where enrichment/tags provide text. Prints artifact
    sizes and child-type counts for budget tracking.

    Returns:
        Process exit code (0 on success).
    """
    enriched = load_enriched()
    ml_tags = load_ml_tags()

    catalog, children = [], []
    type_counts: dict[str, int] = defaultdict(int)
    with open(ML / "movies.csv", newline="") as f:
        for row in csv.DictReader(f):
            mid = int(row["movieId"])
            e = enriched.get(mid, {})
            title, year = clean_title(row["title"])
            catalog.append({
                "id": mid,
                "title": title,
                "year": e.get("year") or year,
                "genres": row["genres"].split("|"),
                "poster": e.get("posterPath"),
                "numRatings": e.get("numRatings", 0),
            })
            for ctype, text in children_for(title, e.get("year") or year, e,
                                            ml_tags.get(mid, [])):
                children.append([mid, ctype, text])
                type_counts[ctype] += 1

    BUILD.mkdir(parents=True, exist_ok=True)
    cat_path = BUILD / "movies.json"
    idx_path = BUILD / "search-index.json"
    cat_path.write_text(json.dumps(catalog, separators=(",", ":")))
    idx_path.write_text(json.dumps({"children": children}, separators=(",", ":")))

    print(f"catalog : {len(catalog)} movies, {cat_path.stat().st_size / 1e6:.2f} MB")
    print(f"children: {len(children)} total -> {idx_path.stat().st_size / 1e6:.2f} MB")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<9} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
