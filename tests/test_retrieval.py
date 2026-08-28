"""Retrieval: BM25 ranking and chunk selection."""

import retrieval


def test_tokenize_lowercases_and_drops_punctuation():
    assert retrieval.tokenize("Hello, World! 42?") == ["hello", "world", "42"]


def test_exact_term_match_ranks_first(prose):
    index = retrieval.ChunkIndex.build(prose)
    assert index.search("quarterly revenue growth", k=3)[0] == 0
    assert index.search("lunar surface Armstrong", k=3)[0] == 1


def test_search_returns_at_most_k(prose):
    """The cap is a ceiling, not a quota: results are never padded up to k."""
    index = retrieval.ChunkIndex.build(prose)
    assert len(index.search("the", k=2)) <= 2
    # k larger than the corpus must not over-return or raise.
    assert len(index.search("the", k=99)) <= len(prose)


def test_query_with_no_matching_term_retrieves_nothing(prose):
    """Returning arbitrary chunks made the QA model answer from page one.

    Every chunk scores zero, the sort leaves them in document order, and the
    first k get read as though they were relevant.
    """
    index = retrieval.ChunkIndex.build(prose)
    assert index.search("zzzz qqqq", k=3) == []
    assert index.search("", k=3) == []


def test_partial_match_returns_only_the_matching_chunks(prose):
    """A query matching one chunk must not be padded out to k."""
    index = retrieval.ChunkIndex.build(prose)
    hits = index.search("photosynthesis", k=5)
    assert hits == [3]


def test_empty_index_returns_nothing():
    index = retrieval.ChunkIndex.build([])
    assert index.search("anything", k=5) == []


def test_idf_penalises_terms_common_to_every_chunk():
    """A term in every document carries no signal and must not dominate."""
    chunks = [f"common word unique{i}" for i in range(5)]
    index = retrieval.ChunkIndex.build(chunks)
    assert index.bm25.idf["common"] < index.bm25.idf["unique3"]


def test_rare_term_beats_repeated_common_term():
    chunks = ["alpha alpha alpha alpha beta", "alpha gamma"]
    index = retrieval.ChunkIndex.build(chunks)
    assert index.search("beta", k=1)[0] == 0
    assert index.search("gamma", k=1)[0] == 1


class TestSelectRepresentative:
    def test_returns_everything_when_under_the_limit(self, prose):
        assert retrieval.select_representative(prose, 24) == list(range(len(prose)))

    def test_never_duplicates_and_stays_in_range(self):
        chunks = [f"chunk {i}" for i in range(60)]
        for limit in (1, 3, 7, 24, 59):
            picked = retrieval.select_representative(chunks, limit)
            assert len(picked) == len(set(picked)), f"duplicates at limit={limit}"
            assert all(0 <= i < len(chunks) for i in picked)
            assert len(picked) <= limit

    def test_returns_document_order(self):
        chunks = [f"chunk {i}" for i in range(60)]
        picked = retrieval.select_representative(chunks, 12)
        assert picked == sorted(picked)

    def test_spans_the_whole_document(self):
        """Selection must sample throughout, not truncate to the first N."""
        chunks = [f"chunk {i}" for i in range(100)]
        picked = retrieval.select_representative(chunks, 10)
        assert picked[0] < 10
        assert picked[-1] > 80

    def test_handles_empty_input(self):
        assert retrieval.select_representative([], 5) == []
