"""PDF text extraction.

Kept free of Streamlit imports so it can be benchmarked and tested directly.

``pypdfium2`` is the primary backend: it binds Google's pdfium, the engine
Chrome uses, and reads text roughly 60x faster than ``pdfplumber``, which builds
a full per-character layout model this app never uses. ``pdfplumber`` is
retained as a fallback for documents pdfium reads as empty.

Parallel extraction was measured and deliberately not adopted; see
benchmarks/README.md for the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

ProgressFn = Callable[[int, int], None]


@dataclass(frozen=True)
class Extraction:
    """The result of reading a PDF."""

    text: str
    page_count: int
    backend: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def _emit(progress: ProgressFn | None, done: int, total: int) -> None:
    if progress is not None:
        progress(done, total)


def extract_pypdfium2(pdf_bytes: bytes, progress: ProgressFn | None = None) -> tuple[str, int]:
    """Extract every page's text with pdfium."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    try:
        total = len(doc)
        # Collect into a list and join once; repeated `str +=` is quadratic.
        parts: list[str] = []
        for index in range(total):
            page = doc[index]
            textpage = page.get_textpage()
            try:
                # get_text_bounded covers the full page with full Unicode support,
                # unlike get_text_range which is limited to UCS-2.
                parts.append(textpage.get_text_bounded())
            finally:
                textpage.close()
                page.close()
            _emit(progress, index + 1, total)
        return "\n".join(parts), total
    finally:
        doc.close()


def extract_pdfplumber(pdf_bytes: bytes, progress: ProgressFn | None = None) -> tuple[str, int]:
    """Extract every page's text with pdfplumber. Slower; used as a fallback."""
    import io

    import pdfplumber

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total = len(pdf.pages)
        parts: list[str] = []
        for index, page in enumerate(pdf.pages):
            parts.append(page.extract_text() or "")
            _emit(progress, index + 1, total)
        return "\n".join(parts), total


def extract(pdf_bytes: bytes, progress: ProgressFn | None = None) -> Extraction:
    """Read `pdf_bytes` with the fastest backend that produces text.

    Falls back to pdfplumber when pdfium returns nothing usable, which can
    happen with unusual font encodings. Neither backend can read a scanned
    document; that needs OCR.
    """
    text, page_count, backend = "", 0, "pypdfium2"
    try:
        text, page_count = extract_pypdfium2(pdf_bytes, progress)
    except Exception:
        text = ""

    if not text.strip():
        try:
            text, page_count = extract_pdfplumber(pdf_bytes, progress)
            backend = "pdfplumber"
        except Exception:
            # Let the caller decide how to surface a document neither backend reads.
            text, backend = "", "none"

    return Extraction(text=text, page_count=page_count, backend=backend)
