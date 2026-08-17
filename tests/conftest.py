"""Shared test fixtures.

Everything here is deliberately free of torch, transformers and model
downloads, so the suite runs in seconds in CI. The handful of behaviours that
genuinely need a model are covered by the scripts in benchmarks/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class WordTokenizer:
    """Stands in for a Hugging Face tokenizer, one token per word.

    `chunk_by_tokens` only needs to measure token counts and decode ids back to
    text, so words work as tokens and the tests need no download.
    """

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [text.split() for text in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


@pytest.fixture
def tokenizer():
    return WordTokenizer()


@pytest.fixture
def prose():
    """Sentences with distinct vocabulary, so retrieval targets are unambiguous."""
    return [
        "The quarterly financial report shows revenue growth across every region.",
        "Neil Armstrong became the first person to walk on the lunar surface.",
        "Shipping containers are stacked according to weight distribution rules.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
        "The ventilation system was replaced during the summer maintenance window.",
    ]


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory):
    """A small real PDF, or skip if the builder is unavailable."""
    reportlab = pytest.importorskip("reportlab", reason="reportlab not installed")
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    assert reportlab  # silence linters

    path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    style = getSampleStyleSheet()["BodyText"]
    paragraphs = [
        Paragraph(
            f"Section {i}. The quick brown fox jumps over the lazy dog. "
            f"This paragraph exists so the extractor has text to find.",
            style,
        )
        for i in range(12)
    ]
    SimpleDocTemplate(str(path), pagesize=LETTER).build(paragraphs)
    return path
