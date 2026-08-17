"""Inference: chunking, the time budget, and the reduce loop.

These cover the logic that does not need a model. Span decoding and batched
generation need torch and are exercised by benchmarks/bench_inference.py.
"""

import time

import inference


class TestChunkByTokens:
    def test_empty_text_yields_no_chunks(self, tokenizer):
        assert inference.chunk_by_tokens("", tokenizer) == []
        assert inference.chunk_by_tokens("   \n  ", tokenizer) == []

    def test_chunks_stay_within_the_token_budget(self, tokenizer):
        text = " ".join(f"Sentence number {i} carries a few words." for i in range(200))
        chunks = inference.chunk_by_tokens(text, tokenizer, budget_tokens=50)
        assert chunks
        for chunk in chunks:
            assert len(chunk.split()) <= 50

    def test_no_empty_chunks(self, tokenizer):
        text = "One. Two. Three. " * 50
        for chunk in inference.chunk_by_tokens(text, tokenizer, budget_tokens=20):
            assert chunk.strip()

    def test_fills_the_window_rather_than_a_fraction_of_it(self, tokenizer):
        """The bug this replaced packed ~25% of the window."""
        text = " ".join(f"Word{i} filler filler filler." for i in range(300))
        chunks = inference.chunk_by_tokens(text, tokenizer, budget_tokens=100)
        # Ignore the final chunk, which is whatever is left over.
        body = chunks[:-1]
        assert body, "expected more than one chunk"
        average = sum(len(c.split()) for c in body) / len(body)
        assert average > 60, f"packing only {average:.0f} of 100 tokens"

    def test_oversized_sentence_is_split_not_dropped(self, tokenizer):
        giant = " ".join(f"word{i}" for i in range(500))  # one sentence, no period
        chunks = inference.chunk_by_tokens(giant, tokenizer, budget_tokens=100)
        assert len(chunks) >= 5
        for chunk in chunks:
            assert len(chunk.split()) <= 100
        # Nothing may be silently lost.
        assert "word0" in chunks[0]
        assert "word499" in chunks[-1]

    def test_overlap_carries_context_between_chunks(self, tokenizer):
        text = " ".join(f"Alpha{i} beta gamma delta epsilon." for i in range(60))
        chunks = inference.chunk_by_tokens(
            text, tokenizer, budget_tokens=40, overlap_tokens=10
        )
        assert len(chunks) > 2
        # The tail of one chunk should reappear at the head of the next.
        first_tail = set(chunks[0].split()[-10:])
        second_head = set(chunks[1].split()[:10])
        assert first_tail & second_head

    def test_zero_overlap_produces_no_repetition(self, tokenizer):
        text = " ".join(f"Alpha{i} beta gamma." for i in range(60))
        chunks = inference.chunk_by_tokens(
            text, tokenizer, budget_tokens=30, overlap_tokens=0
        )
        assert len(chunks) > 2
        assert not set(chunks[0].split()[-5:]) & set(chunks[1].split()[:5])


class TestBudget:
    def test_none_never_expires(self):
        assert inference.Budget(None).expired is False

    def test_zero_is_immediately_expired(self):
        assert inference.Budget(0.0).expired is True

    def test_unexpired_within_its_window(self):
        assert inference.Budget(30).expired is False

    def test_elapsed_advances(self):
        budget = inference.Budget(30)
        time.sleep(0.01)
        assert budget.elapsed > 0


class TestReduceSummaries:
    def test_empty_input(self, tokenizer):
        assert inference.reduce_summaries([], tokenizer, object()) == ""

    def test_single_summary_passes_through(self, tokenizer):
        assert inference.reduce_summaries(["only one"], tokenizer, object()) == "only one"

    def test_short_input_is_not_folded(self, tokenizer, monkeypatch):
        """Already under target: the model must not be called at all."""
        calls = []

        def spy(pieces, *args, **kwargs):
            calls.append(pieces)
            return list(pieces)

        monkeypatch.setattr(inference, "summarize_chunks", spy)
        result = inference.reduce_summaries(["short one", "short two"], tokenizer, object())
        assert calls == []
        assert result == "short one short two"

    def test_folds_long_input_down(self, tokenizer, monkeypatch):
        monkeypatch.setattr(
            inference,
            "summarize_chunks",
            lambda pieces, *a, **k: [" ".join(p.split()[:5]) for p in pieces],
        )
        long_parts = [" ".join(f"word{i}" for i in range(400)) for _ in range(6)]
        result = inference.reduce_summaries(long_parts, tokenizer, object())
        assert len(result) < sum(len(p) for p in long_parts)

    def test_terminates_when_folding_stops_shrinking(self, tokenizer, monkeypatch):
        """A model that returns more text than it was given must not loop."""
        monkeypatch.setattr(
            inference,
            "summarize_chunks",
            lambda pieces, *a, **k: [p + " and then some extra padding" for p in pieces],
        )
        long_parts = [" ".join(f"word{i}" for i in range(400)) for _ in range(4)]
        result = inference.reduce_summaries(long_parts, tokenizer, object())
        assert isinstance(result, str) and result

    def test_expired_budget_stops_folding(self, tokenizer, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("summarize_chunks called despite an expired budget")

        monkeypatch.setattr(inference, "summarize_chunks", fail)

        long_parts = [" ".join(f"word{i}" for i in range(400)) for _ in range(6)]
        result = inference.reduce_summaries(
            long_parts, tokenizer, object(), budget=inference.Budget(0.0)
        )
        assert isinstance(result, str)


def test_split_sentences_falls_back_without_nltk(monkeypatch):
    """The naive split must cover deployments where NLTK data is missing."""
    monkeypatch.setitem(__import__("sys").modules, "nltk.tokenize", None)
    sentences = inference.split_sentences("One thing. Two things. Three things.")
    assert len(sentences) >= 3
