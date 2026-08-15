"""Measure Stage 3 retrieval: recall, answer agreement, and speed.

Retrieval quality is measured without hand-labelled data. A sentence is drawn
from a known chunk and used as the query; retrieval should return the chunk it
came from. That gives a recall@k figure on any document.

The figure that actually matters for this change is *agreement*: does answering
from the top-k chunks produce the same answer as scanning every chunk? Speed is
worthless if the answers get worse.

Usage:
    python benchmarks/bench_retrieval.py benchmarks/fixture_60p.pdf --questions 12
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference  # noqa: E402
import pdf_extract  # noqa: E402
import retrieval  # noqa: E402


def sample_queries(chunks, count, seed=0):
    """Draw sentences from known chunks to use as queries."""
    rng = random.Random(seed)
    pool = []
    for index, chunk in enumerate(chunks):
        for sentence in inference.split_sentences(chunk):
            words = sentence.split()
            # Long enough to be a meaningful query, short enough to be one.
            if 8 <= len(words) <= 40:
                pool.append((sentence.strip(), index))
    rng.shuffle(pool)
    return pool[:count]


def recall_at_k(index, queries, k, mode):
    """Fraction of queries whose source chunk is in the top k."""
    import numpy as np

    hits = 0
    for query, want in queries:
        if mode == "bm25":
            scores = index.bm25.scores(query)
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        elif mode == "dense":
            if not index.dense_enabled:
                return float("nan")
            vector = np.asarray(index.embedder.encode([query]), dtype="float32")[0]
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
            ranked = [int(i) for i in np.argsort(-(index.embeddings @ vector))[:k]]
        else:
            ranked = index.search(query, k=k)
        hits += want in ranked
    return hits / len(queries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--questions", type=int, default=12)
    ap.add_argument("--k", type=int, default=retrieval.TOP_K)
    ap.add_argument("--chars", type=int, default=0, help="truncate the document")
    args = ap.parse_args()

    text = pdf_extract.extract(args.pdf.read_bytes()).text
    if args.chars:
        text = text[: args.chars]
    print(f"document: {len(text)} chars")

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    qa_tok = AutoTokenizer.from_pretrained(inference.QA_MODEL)
    qa_model = inference.prepare(
        AutoModelForQuestionAnswering.from_pretrained(inference.QA_MODEL)
    )

    chunks = inference.chunk_by_tokens(text, qa_tok, inference.QA_INPUT_TOKENS)
    print(f"chunks: {len(chunks)}")

    t0 = time.perf_counter()
    embedder = retrieval.load_embedder()
    load_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    index = retrieval.ChunkIndex.build(chunks, embedder)
    build_time = time.perf_counter() - t0
    print(f"embedder load: {load_time:.2f}s   index build: {build_time * 1000:.1f}ms   "
          f"dense: {index.dense_enabled}\n")

    queries = sample_queries(chunks, args.questions)
    if not queries:
        print("no usable queries in this document")
        return

    print(f"recall@{args.k} over {len(queries)} queries")
    for mode in ("bm25", "dense", "hybrid"):
        print(f"  {mode:<8}{recall_at_k(index, queries, args.k, mode):.0%}")

    # Answer agreement and speed: exhaustive scan vs retrieve-then-read.
    print(f"\nexhaustive scan vs top-{args.k}")
    agree = 0
    full_total = fast_total = 0.0

    for query, _ in queries:
        t0 = time.perf_counter()
        full = inference.answer_from_chunks(query, chunks, qa_tok, qa_model)
        full_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        picked = index.search(query, k=args.k)
        fast = inference.answer_from_chunks(
            query, [chunks[i] for i in picked], qa_tok, qa_model
        )
        fast_total += time.perf_counter() - t0

        agree += (full.text if full else "") == (fast.text if fast else "")

    n = len(queries)
    print(f"  answers identical: {agree}/{n} ({agree / n:.0%})")
    print(f"  full scan:  {full_total / n:.3f}s per question")
    print(f"  retrieved:  {fast_total / n:.3f}s per question")
    print(f"  speedup:    {full_total / fast_total:.1f}x")

    # Summary selection coverage: does spreading beat even spacing?
    limit = inference.SUMMARY_MAX_MAP_CHUNKS
    if index.dense_enabled and len(chunks) > limit:
        spread = retrieval.select_representative(chunks, limit, index)
        step = len(chunks) / limit
        even = sorted({min(len(chunks) - 1, int(i * step)) for i in range(limit)})
        print(f"\nsummary selection ({limit} of {len(chunks)} chunks), mean similarity "
              f"to nearest selected")
        print(f"  evenly spaced: {retrieval.coverage(chunks, even, index):.4f}")
        print(f"  spread (MMR):  {retrieval.coverage(chunks, spread, index):.4f}")


if __name__ == "__main__":
    main()
