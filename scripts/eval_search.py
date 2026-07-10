#!/usr/bin/env python3
"""Step-4 search eval: golden queries -> per-leg recall@9 / MRR / NDCG@9.

Mirrors the browser pipeline component-for-component: same BM25 (tokenizer,
k1=1.2, b=0.75, Lucene IDF floor), same bge-small query prefix, same RRF
k=60 + best-hit grouping, same cross-encoder (ms-marco-MiniLM-L-6-v2 —
PyTorch here, quantized ONNX in the browser; scores differ in the 3rd
decimal, ordering is what matters). Dense goes straight to Pinecone
rather than through the Worker (identical results, no proxy hop).

Legs reported: dense-only, bm25-only, fused (RRF), fused+rerank.
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import dotenv_values
from pinecone import Pinecone
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "data" / "build"

K1, B = 1.2, 0.75
RRF_K = 60
LEG_TOP_K = 60
GROUP_TOP_N = 20
AT = 9  # grid size — the user sees 9 posters
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize exactly like web/src/bm25.ts: lowercase alnum runs.

    Args:
        text: query or child text.

    Returns:
        Ordered token list (duplicates preserved).
    """
    return TOKEN_RE.findall(text.lower())


class Bm25:
    """Inverted-index BM25 mirroring the browser implementation.

    Built once over all children; scores queries with deduped terms and
    the Lucene non-negative IDF, identical constants to bm25.ts.
    """

    def __init__(self, texts: list[str]):
        """Index the corpus.

        Args:
            texts: child texts in id order (index = child id).
        """
        self.n = len(texts)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = [0] * self.n
        total = 0
        for i, t in enumerate(texts):
            toks = tokenize(t)
            self.doc_len[i] = len(toks)
            total += len(toks)
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            for term, f in tf.items():
                self.postings[term].append((i, f))
        self.avg_len = total / self.n

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Score a query and return the top_k (child_id, score) hits.

        Args:
            query: raw query text.
            top_k: hits to return.

        Returns:
            (child_id, score) pairs sorted by score desc.
        """
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                norm = tf / (tf + K1 * (1 - B + B * self.doc_len[i] / self.avg_len))
                scores[i] += idf * (K1 + 1) * norm
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


def resolve_titles(golden: dict, catalog: list[dict]) -> tuple[list[dict], list[str]]:
    """Resolve golden-set title strings to movieIds against the catalog.

    Exact case-insensitive title match first; unique substring match as
    fallback. Ambiguous or missing titles are collected as errors so the
    golden set can be corrected rather than silently mis-scored.

    Args:
        golden: parsed golden-queries.json.
        catalog: movies.json records.

    Returns:
        (queries, errors): queries carry rel = {movieId: grade}; errors
        are human-readable resolution failures.
    """
    by_title: dict[str, list[int]] = defaultdict(list)
    for m in catalog:
        by_title[m["title"].lower()].append(m["id"])

    def find(title: str) -> int | None:
        ids = by_title.get(title.lower())
        if ids and len(ids) >= 1:
            return ids[0]  # duplicates: earliest catalog entry wins
        subs = [mid for t, idlist in by_title.items() if title.lower() in t for mid in idlist]
        return subs[0] if len(subs) == 1 else None

    queries, errors = [], []
    for q in golden["queries"]:
        rel: dict[int, int] = {}
        for grade, key in ((2, "rel2"), (1, "rel1")):
            for title in q[key]:
                mid = find(title)
                if mid is None:
                    errors.append(f"{q['q']!r}: cannot resolve {title!r}")
                else:
                    rel[mid] = max(rel.get(mid, 0), grade)
        queries.append({"type": q["type"], "q": q["q"], "rel": rel})
    return queries, errors


def group_best(hits: list[tuple[int, float]], child_parent: list[int]) -> list[int]:
    """Best-hit group child hits to unique parent movieIds (max, not sum).

    Args:
        hits: (child_id, score) sorted desc.
        child_parent: child id -> parent movieId.

    Returns:
        Parent movieIds in first-seen (= best-hit) order, top GROUP_TOP_N.
    """
    seen: list[int] = []
    for cid, _ in hits:
        mid = child_parent[cid]
        if mid not in seen:
            seen.append(mid)
        if len(seen) >= GROUP_TOP_N:
            break
    return seen


def metrics(ranked: list[int], rel: dict[int, int]) -> tuple[float, float, float]:
    """Compute recall@AT, MRR@AT and graded NDCG@AT for one query.

    Args:
        ranked: movieIds best-first.
        rel: movieId -> grade (2 or 1).

    Returns:
        (recall, mrr, ndcg) at cutoff AT. Recall counts any graded item;
        NDCG uses the grades as gains.
    """
    top = ranked[:AT]
    hits = sum(1 for m in top if m in rel)
    recall = hits / min(len(rel), AT) if rel else 0.0
    mrr = 0.0
    for i, m in enumerate(top):
        if m in rel:
            mrr = 1 / (i + 1)
            break
    dcg = sum(rel.get(m, 0) / math.log2(i + 2) for i, m in enumerate(top))
    ideal = sorted(rel.values(), reverse=True)[:AT]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return recall, mrr, (dcg / idcg if idcg else 0.0)


def main() -> int:
    """Run all four pipeline variants over the golden set; print tables.

    Progress to stderr, final markdown (overall + per-query-type) to
    stdout for pasting into EVALUATION.md.

    Returns:
        0 on success, 1 on unresolvable golden-set titles (fix and rerun).
    """
    catalog = json.loads((BUILD / "movies.json").read_text())
    children = json.loads((BUILD / "search-index.json").read_text())["children"]
    child_parent = [c[0] for c in children]
    golden = json.loads((ROOT / "data" / "golden-queries.json").read_text())

    queries, errors = resolve_titles(golden, catalog)
    if errors:
        print("TITLE RESOLUTION FAILURES:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(f"{len(queries)} queries resolved clean", file=sys.stderr)

    print("building BM25...", file=sys.stderr)
    bm25 = Bm25([c[2] for c in children])

    print("embedding queries...", file=sys.stderr)
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    qvecs = embedder.encode([QUERY_PREFIX + q["q"] for q in queries],
                            normalize_embeddings=True)

    print("querying Pinecone...", file=sys.stderr)
    index = Pinecone(api_key=dotenv_values(ROOT / ".env")["PINECONE_API_KEY"]).Index("moviebro")
    dense_hits = []
    for vec in qvecs:
        out = index.query(vector=vec.tolist(), top_k=LEG_TOP_K)
        dense_hits.append([(int(m["id"]), m["score"]) for m in out["matches"]])

    print("loading cross-encoder...", file=sys.stderr)
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    title_of = {m["id"]: (m["title"], m["year"]) for m in catalog}
    overview: dict[int, str] = defaultdict(str)
    extra: dict[int, list[str]] = defaultdict(list)  # cast + keywords + genres-ish
    for mid, ctype, text in children:
        if ctype == "overview":
            overview[mid] = f"{overview[mid]} {text}".strip()
        elif ctype in ("cast", "keywords"):
            extra[mid].append(text)

    def doc_for(mid: int) -> str:
        """Rerank document, identical construction to web/src/rerank.ts.

        Carries the SAME evidence the retrieval legs matched on — title,
        overview, cast, keywords. First eval round used title+overview
        only and rerank DEMOTED actor/constraint hits it couldn't see
        (recall@9 0.45 -> 0.40); see EVALUATION.md finding.
        """
        t, y = title_of[mid]
        head = f"{t} ({y})" if y else t
        parts = [overview.get(mid, ""), *extra.get(mid, [])]
        return f"{head}. {' '.join(p for p in parts if p)}"

    def rrf(dense: list[tuple[int, float]], bm: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """RRF k=60 child-level fusion, both legs, sorted desc."""
        acc: dict[int, float] = defaultdict(float)
        for hits in (dense, bm):
            for rank, (cid, _) in enumerate(hits):
                acc[cid] += 1 / (RRF_K + rank + 1)
        return sorted(acc.items(), key=lambda x: -x[1])

    legs: dict[str, list[list[int]]] = {"dense-only": [], "bm25-only": [], "fused (RRF k=60)": [], "fused + rerank": []}
    print("running pipelines...", file=sys.stderr)
    for qi, q in enumerate(queries):
        bm_hits = bm25.search(q["q"], LEG_TOP_K)
        legs["dense-only"].append(group_best(dense_hits[qi], child_parent))
        legs["bm25-only"].append(group_best(bm_hits, child_parent))
        fused = group_best(rrf(dense_hits[qi], bm_hits), child_parent)
        legs["fused (RRF k=60)"].append(fused)
        pairs = [(q["q"], doc_for(mid)) for mid in fused]
        scores = ce.predict(pairs)
        legs["fused + rerank"].append([mid for mid, _ in sorted(zip(fused, scores), key=lambda x: -x[1])])

    # ---- report ----
    print(f"\n| Pipeline | recall@{AT} | MRR@{AT} | NDCG@{AT} |")
    print("|---|---|---|---|")
    for name, ranked_all in legs.items():
        r, m, n = zip(*(metrics(rk, q["rel"]) for rk, q in zip(ranked_all, queries)))
        print(f"| {name} | {sum(r)/len(r):.4f} | {sum(m)/len(m):.4f} | {sum(n)/len(n):.4f} |")

    print(f"\nPer query type (fused + rerank):")
    print(f"\n| Type | n | recall@{AT} | MRR@{AT} | NDCG@{AT} |")
    print("|---|---|---|---|---|")
    for qtype in ("title", "actor", "vibe", "constraint"):
        idx = [i for i, q in enumerate(queries) if q["type"] == qtype]
        r, m, n = zip(*(metrics(legs["fused + rerank"][i], queries[i]["rel"]) for i in idx))
        print(f"| {qtype} | {len(idx)} | {sum(r)/len(r):.4f} | {sum(m)/len(m):.4f} | {sum(n)/len(n):.4f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
