"""PDF extraction: the pdfium path, the fallback, and progress reporting."""

import pytest

import pdf_extract

pytest.importorskip("pypdfium2", reason="pypdfium2 not installed")


def test_extracts_text_and_page_count(sample_pdf):
    result = pdf_extract.extract(sample_pdf.read_bytes())
    assert result.page_count >= 1
    assert "quick brown fox" in result.text
    assert result.char_count == len(result.text)
    assert result.backend == "pypdfium2"


def test_every_page_contributes_text(sample_pdf):
    result = pdf_extract.extract(sample_pdf.read_bytes())
    assert "Section 0." in result.text
    assert "Section 11." in result.text


def test_progress_is_reported_to_completion(sample_pdf):
    seen = []
    result = pdf_extract.extract(
        sample_pdf.read_bytes(), progress=lambda done, total: seen.append((done, total))
    )
    assert seen, "no progress was reported"
    assert seen[-1] == (result.page_count, result.page_count)
    # Monotonic, and never over-reporting.
    assert all(a <= b for (a, _), (b, _) in zip(seen, seen[1:]))
    assert all(done <= total for done, total in seen)


def test_extraction_works_without_a_progress_callback(sample_pdf):
    assert pdf_extract.extract(sample_pdf.read_bytes()).text


def test_unreadable_input_reports_no_backend_rather_than_raising():
    """Both backends failing is a normal outcome the caller has to render.

    The app turns this into an "is it a scan?" message, so extract() reports
    it rather than raising.
    """
    result = pdf_extract.extract(b"this is not a pdf")
    assert result.text == ""
    assert result.backend == "none"
    assert not result.text.strip()


def test_backend_names_the_path_actually_used(sample_pdf):
    """The status line shows this, so it has to be truthful."""
    assert pdf_extract.extract(sample_pdf.read_bytes()).backend in {
        "pypdfium2",
        "pdfplumber",
    }
