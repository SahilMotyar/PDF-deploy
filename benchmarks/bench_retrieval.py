"""Measure Stage 3 retrieval: recall, answer correctness, and speed.

Retrieval quality is measured without hand-labelled data. A sentence is drawn
from a known chunk and used as the query; retrieval should return the chunk it
came from. That gives a recall@k figure on any document.

The figure that matters most is where the *answer* comes from. Because each
query is lifted from a known chunk, the answer should be drawn from that chunk.
Comparing the exhaustive scan against retrieve-then-read on that basis shows
whether reading fewer chunks costs anything. It does not -- it helps.

Usage:
    python benchmarks/bench_retrieval.py benchmarks/fixture_60p.pdf --questions 20
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


def degrade(query, keep=0.4, seed=0):
    """Keep a shuffled subset of the longer words, approximating a rephrasing."""
    rng = random.Random(seed)
    words = [w for w in query.split() if len(w) > 3]
    if len(words) < 3:
        return query
    picked = rng.sample(words, max(3, int(len(words) * keep)))
    rng.shuffle(picked)
    return " ".join(picked)


def recall_at_k(index, queries, k):
    """Fraction of queries whose source chunk is in the top k."""
    hits = sum(want in index.search(query, k=k) for query, want in queries)
    return hits / len(queries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--questions", type=int, default=20)
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
    index = retrieval.ChunkIndex.build(chunks)
    print(f"index build: {(time.perf_counter() - t0) * 1000:.1f}ms\n")

    queries = sample_queries(chunks, args.questions)
    if not queries:
        print("no usable queries in this document")
        return

    # Verbatim sentences are the ideal case for lexical matching, so also test
    # degraded queries -- a subset of the content words, shuffled -- which is
    # closer to how someone actually phrases a question.
    partial = [(degrade(q, seed=i), want) for i, (q, want) in enumerate(queries)]

    print(f"recall@{args.k} over {len(queries)} queries")
    print(f"  verbatim wording: {recall_at_k(index, queries, args.k):.0%}")
    print(f"  partial wording:  {recall_at_k(index, partial, args.k):.0%}")

    print(f"\nexhaustive scan vs top-{args.k}")
    agree = full_right = fast_right = 0
    full_total = fast_total = 0.0

    for query, want in queries:
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
        full_right += full is not None and full.chunk_index == want
        # fast.chunk_index indexes into the retrieved subset, not the document.
        fast_right += fast is not None and picked[fast.chunk_index] == want

    n = len(queries)
    print("  answer drawn from the correct chunk:")
    print(f"    full scan:  {full_right}/{n} ({full_right / n:.0%})")
    print(f"    retrieved:  {fast_right}/{n} ({fast_right / n:.0%})")
    print(f"  answers identical to full scan: {agree}/{n} ({agree / n:.0%})")
    print(f"  full scan:  {full_total / n:.3f}s per question")
    print(f"  retrieved:  {fast_total / n:.3f}s per question")
    print(f"  speedup:    {full_total / fast_total:.1f}x")


if __name__ == "__main__":
    main()
