"""Chunk retrieval, so Q&A stops scanning the whole document.

The app previously ran a DistilBERT forward pass over *every* chunk for every
question. This narrows that to the handful of chunks worth reading, which is
the difference between O(document) and O(k) per question.

It also makes the answers better, which was not the goal. Scoring spans across
sixty-odd chunks gives the QA model sixty-odd chances to return a confident
span from text that has nothing to do with the question, and the "best score
wins" loop then picks one. Reading five chunks instead of sixty-five raised the
share of answers drawn from the correct chunk from 25% to 65%.

Retrieval is BM25: lexical, pure Python, no model and no download. Static
embeddings (model2vec) were tried alongside it and measurably *hurt* -- see
`benchmarks/README.md` for the numbers and the caveat on how they were
obtained.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

# Chunks handed to the QA model per question. Five 440-token chunks is roughly
# 2,200 tokens of context, which comfortably covers a fact and its surroundings.
TOP_K = 5

# BM25 constants, the standard defaults.
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over an inverted index."""

    def __init__(self, documents: Sequence[str]):
        self.n = len(documents)
        self.lengths = [0] * self.n
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for i, document in enumerate(documents):
            terms = tokenize(document)
            self.lengths[i] = len(terms)
            for term, freq in Counter(terms).items():
                self.postings[term].append((i, freq))

        self.avg_length = (sum(self.lengths) / self.n) if self.n else 0.0
        self.idf = {
            term: math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def scores(self, query: str) -> list[float]:
        """BM25 score for every document. Only posting lists for query terms
        are touched, so cost tracks the query, not the document."""
        out = [0.0] * self.n
        if not self.avg_length:
            return out

        for term in tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for doc, freq in posting:
                norm = 1 - BM25_B + BM25_B * (self.lengths[doc] / self.avg_length)
                out[doc] += idf * (freq * (BM25_K1 + 1)) / (freq + BM25_K1 * norm)
        return out


@dataclass
class ChunkIndex:
    """A searchable index over one document's chunks."""

    chunks: list[str]
    bm25: BM25

    @classmethod
    def build(cls, chunks: Sequence[str]) -> "ChunkIndex":
        chunks = list(chunks)
        return cls(chunks=chunks, bm25=BM25(chunks))

    def search(self, query: str, k: int = TOP_K) -> list[int]:
        """Return chunk indices for the k most promising chunks, best first."""
        if not self.chunks:
            return []

        scores = self.bm25.scores(query)
        ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        return ranked[: min(k, len(self.chunks))]


def select_representative(chunks: Sequence[str], limit: int) -> list[int]:
    """Choose at most `limit` chunks, evenly spaced through the document.

    Embedding-based selection was tried here and did not earn its complexity.
    Measured against 65 chunks of real prose by mean similarity of every chunk
    to its nearest selected one, at the limit of 24 this actually uses:

        evenly spaced           0.8781
        k-means centroid        0.8775
        max-marginal-relevance  0.8736

    The spread is under half a percent, and max-marginal-relevance is the worst
    of the three because maximising diversity selects outliers, which is close
    to the opposite of choosing representative text. Even spacing is also the
    only one of the three that needs no embedding model, and it returns
    document order for free.
    """
    total = len(chunks)
    if total <= limit:
        return list(range(total))

    step = total / limit
    return sorted({min(total - 1, int(i * step)) for i in range(limit)})
