"""Chunk retrieval, so Q&A stops scanning the whole document.

The app previously ran a DistilBERT forward pass over *every* chunk for every
question. This narrows that to the handful of chunks worth reading, which is
the difference between O(document) and O(k) per question.

Two signals are combined:

* **BM25**, lexical, pure Python. Free, no download, and hard to beat when the
  question shares vocabulary with the passage -- names, numbers, identifiers.
* **Static embeddings** via model2vec, which handles paraphrase ("walked on the
  Moon" against "lunar surface"). These are token lookups plus pooling rather
  than a transformer forward pass, so encoding a whole document costs
  milliseconds.

The dense half is optional. If model2vec is missing or its weights cannot be
fetched, retrieval degrades to BM25 alone rather than failing -- worth having on
a deploy target where the model download may not be available.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

EMBEDDING_MODEL = "minishlab/potion-base-8M"

# Chunks handed to the QA model per question. Five 440-token chunks is roughly
# 2,200 tokens of context, which comfortably covers a fact and its surroundings.
TOP_K = 5

# BM25 constants, the standard defaults.
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal-rank-fusion damping. Fusing on rank rather than raw score avoids
# having to calibrate two scales against each other.
RRF_K = 60

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


def load_embedder(model_name: str = EMBEDDING_MODEL):
    """Load the static embedding model, or None if unavailable."""
    try:
        from model2vec import StaticModel

        return StaticModel.from_pretrained(model_name)
    except Exception:
        return None


def _rrf(ranked: Sequence[int], weights: dict[int, float], weight: float = 1.0) -> None:
    for rank, index in enumerate(ranked):
        weights[index] = weights.get(index, 0.0) + weight / (RRF_K + rank + 1)


@dataclass
class ChunkIndex:
    """A searchable index over one document's chunks."""

    chunks: list[str]
    bm25: BM25
    embedder: object | None = None
    embeddings: object | None = None  # numpy array, or None
    _order: list[int] = field(default_factory=list)

    @classmethod
    def build(cls, chunks: Sequence[str], embedder=None) -> "ChunkIndex":
        chunks = list(chunks)
        embeddings = None
        if embedder is not None and chunks:
            try:
                import numpy as np

                vectors = np.asarray(embedder.encode(chunks), dtype="float32")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                embeddings = vectors / np.clip(norms, 1e-9, None)
            except Exception:
                embeddings = None
        return cls(chunks=chunks, bm25=BM25(chunks), embedder=embedder, embeddings=embeddings)

    @property
    def dense_enabled(self) -> bool:
        return self.embeddings is not None

    def search(self, query: str, k: int = TOP_K) -> list[int]:
        """Return chunk indices for the k most promising chunks, best first."""
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))

        lexical = self.bm25.scores(query)
        lexical_rank = sorted(range(len(self.chunks)), key=lambda i: -lexical[i])

        fused: dict[int, float] = {}
        _rrf(lexical_rank[: k * 4], fused)

        if self.embeddings is not None:
            try:
                import numpy as np

                vector = np.asarray(self.embedder.encode([query]), dtype="float32")[0]
                vector /= max(float(np.linalg.norm(vector)), 1e-9)
                similarity = self.embeddings @ vector
                dense_rank = list(np.argsort(-similarity)[: k * 4])
                _rrf([int(i) for i in dense_rank], fused)
            except Exception:
                pass  # lexical results stand on their own

        return sorted(fused, key=lambda i: -fused[i])[:k]


def select_representative(
    chunks: Sequence[str], limit: int, index: ChunkIndex | None = None
) -> list[int]:
    """Choose at most `limit` chunks that cover the document.

    With embeddings, this is greedy maximal-marginal-relevance: repeatedly take
    the chunk least similar to everything already chosen, which spreads the
    selection across topics instead of over-sampling a repetitive section.
    Without them it falls back to even spacing. Either way the result is
    returned in document order, so the summary still reads front to back.
    """
    total = len(chunks)
    if total <= limit:
        return list(range(total))

    if index is not None and index.dense_enabled:
        try:
            import numpy as np

            vectors = index.embeddings
            # Start from the chunk closest to the document centroid: the most
            # representative single chunk.
            centroid = vectors.mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
            chosen = [int(np.argmax(vectors @ centroid))]

            # Track each candidate's similarity to the nearest chosen chunk.
            nearest = vectors @ vectors[chosen[0]]
            for _ in range(limit - 1):
                nearest[chosen[-1]] = np.inf  # never reselect
                pick = int(np.argmin(nearest))
                chosen.append(pick)
                nearest = np.minimum(nearest, vectors @ vectors[pick])

            return sorted(chosen)
        except Exception:
            pass

    step = total / limit
    return sorted({min(total - 1, int(i * step)) for i in range(limit)})


def coverage(chunks: Sequence[str], selected: Sequence[int], index: ChunkIndex) -> float:
    """Mean similarity of every chunk to its closest selected chunk.

    A measurable proxy for "does this selection represent the document": 1.0
    would mean every chunk is perfectly covered by something chosen.
    """
    if not index.dense_enabled or not selected:
        return float("nan")

    import numpy as np

    vectors = index.embeddings
    return float((vectors @ vectors[list(selected)].T).max(axis=1).mean())
