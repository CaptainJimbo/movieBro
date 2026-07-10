#!/usr/bin/env python3
"""Embed all search children with bge-small-en-v1.5 and upsert to Pinecone.

The bge discipline (same as limenarchisAI): PASSAGES are embedded with no
prefix; QUERIES get "Represent this sentence for searching relevant
passages: " — the browser side must apply that query prefix. Vectors are
L2-normalized so Pinecone's cosine metric behaves.

Vector id = the child's position in search-index.json's children array, so
the browser can map Pinecone matches straight back to child text with zero
extra metadata shipped.
"""

import json
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "data" / "build"
INDEX_NAME = "moviebro"
BATCH_EMBED = 256
BATCH_UPSERT = 200


def load_children() -> list[list]:
    """Load the child list produced by build_index.py.

    Returns:
        The [[movieId, type, text], ...] array; a child's list index is its
        canonical id everywhere (browser, Pinecone, this script).
    """
    return json.loads((BUILD / "search-index.json").read_text())["children"]


def embed_all(texts: list[str]) -> "np.ndarray":
    """Embed passages with bge-small-en-v1.5, normalized, no prefix.

    Args:
        texts: raw child texts (passage side of the bge asymmetry).

    Returns:
        float32 array [n, 384], L2-normalized rows.
    """
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return model.encode(texts, batch_size=BATCH_EMBED, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)


def main() -> int:
    """Embed every child and upsert [id, vector, {m: movieId}] to Pinecone.

    Progress prints every ~10 batches. Metadata carries only the parent
    movieId (int) — handy for server-side debugging; the browser resolves
    everything else locally from search-index.json.

    Returns:
        Process exit code (0 on success, 1 if PINECONE_API_KEY missing).
    """
    key = dotenv_values(ROOT / ".env").get("PINECONE_API_KEY")
    if not key:
        print("PINECONE_API_KEY missing", file=sys.stderr)
        return 1

    children = load_children()
    texts = [c[2] for c in children]
    print(f"embedding {len(texts)} children...", flush=True)
    t0 = time.time()
    vecs = embed_all(texts)
    print(f"embedded in {time.time() - t0:.0f}s -> {vecs.shape}", flush=True)

    index = Pinecone(api_key=key).Index(INDEX_NAME)
    print("upserting to Pinecone...", flush=True)
    t0 = time.time()
    for start in range(0, len(children), BATCH_UPSERT):
        batch = [
            {"id": str(i), "values": vecs[i].tolist(),
             "metadata": {"m": children[i][0]}}
            for i in range(start, min(start + BATCH_UPSERT, len(children)))
        ]
        index.upsert(vectors=batch)
        if (start // BATCH_UPSERT) % 10 == 0:
            print(f"  upserted {start + len(batch)}/{len(children)}", flush=True)

    print(f"upsert done in {time.time() - t0:.0f}s")
    print(index.describe_index_stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
