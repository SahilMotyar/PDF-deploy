"""The Streamlit app module: document loading and per-document state.

These need streamlit installed, which is why requirements-dev.txt carries it.
They do not need torch: read_pdf never touches a model.
"""

import io

import pytest

pytest.importorskip("streamlit", reason="streamlit not installed")

import PDFread  # noqa: E402


class TestReadPdf:
    """A failed load must be distinguishable from a successful one.

    Both used to return a plain string, so the caller announced "No text could
    be extracted" with st.success() and unlocked both tabs anyway.
    """

    def test_valid_pdf_reports_success(self, sample_pdf):
        result = PDFread.PDFAssistant().read_pdf(io.BytesIO(sample_pdf.read_bytes()))
        assert result.ok is True
        assert "loaded successfully" in result.message

    @pytest.mark.parametrize(
        "data, label",
        [(b"this is not a pdf", "garbage"), (b"", "empty"), (b"%PDF-1.4\n", "truncated")],
    )
    def test_unreadable_input_reports_failure(self, data, label):
        result = PDFread.PDFAssistant().read_pdf(io.BytesIO(data))
        assert result.ok is False, f"{label} was reported as a successful load"
        assert result.message

    def test_text_is_extracted_onto_the_assistant(self, sample_pdf):
        assistant = PDFread.PDFAssistant()
        assistant.read_pdf(io.BytesIO(sample_pdf.read_bytes()))
        assert "quick brown fox" in assistant.pdf_text

    def test_loading_clears_state_from_the_previous_document(self, sample_pdf):
        assistant = PDFread.PDFAssistant()
        assistant.summary = "summary of an earlier document"
        assistant._chunk_cache["qa"] = ["stale chunk"]

        assistant.read_pdf(io.BytesIO(sample_pdf.read_bytes()))

        assert assistant.summary == ""
        assert assistant._chunk_cache == {}

    def test_failed_load_does_not_keep_the_old_document(self, sample_pdf):
        """A failed load must not leave the previous document readable."""
        assistant = PDFread.PDFAssistant()
        assistant.read_pdf(io.BytesIO(sample_pdf.read_bytes()))
        assert assistant.pdf_text

        result = assistant.read_pdf(io.BytesIO(b"not a pdf"))
        assert result.ok is False
        assert not assistant.pdf_text.strip()


class TestClearDocumentState:
    KEYS = ("summary", "conversation", "last_question", "last_answer")

    def test_removes_everything_derived_from_the_document(self):
        for key in self.KEYS:
            PDFread.st.session_state[key] = f"value for {key}"

        PDFread.clear_document_state()

        for key in self.KEYS:
            assert key not in PDFread.st.session_state, f"{key} survived"

    def test_is_safe_when_nothing_is_set(self):
        PDFread.clear_document_state()
        PDFread.clear_document_state()  # idempotent

    def test_leaves_unrelated_state_alone(self):
        PDFread.st.session_state["assistant"] = "keep me"
        PDFread.st.session_state["summary"] = "drop me"

        PDFread.clear_document_state()

        assert PDFread.st.session_state["assistant"] == "keep me"
        assert "summary" not in PDFread.st.session_state
