"""Compare PDF text-extraction backends on a fixture document.

Usage:
    python benchmarks/make_fixture.py --pages 60 --out benchmarks/fixture_60p.pdf
    python benchmarks/bench_extract.py benchmarks/fixture_60p.pdf --repeat 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdf_extract  # noqa: E402


def time_it(fn, repeat: int) -> tuple[float, float, str]:
    """Return (best seconds, median seconds, a checksum of the text)."""
    timings = []
    text = ""
    for _ in range(repeat):
        start = time.perf_counter()
        text, _pages = fn()
        timings.append(time.perf_counter() - start)
    checksum = f"{len(text)}c"
    return min(timings), statistics.median(timings), checksum


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    pdf_bytes = args.pdf.read_bytes()

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    pages = len(doc)
    doc.close()

    print(f"{args.pdf.name}: {pages} pages, {len(pdf_bytes) / 1024:.0f} KiB, repeat={args.repeat}\n")

    cases = [
        ("pdfplumber (baseline)", lambda: pdf_extract.extract_pdfplumber(pdf_bytes)),
        ("pypdfium2", lambda: pdf_extract.extract_pypdfium2(pdf_bytes)),
    ]

    print(f"{'backend':<26}{'best':>10}{'median':>10}{'chars':>12}{'speedup':>10}")
    print("-" * 68)

    baseline = None
    for name, fn in cases:
        best, median, checksum = time_it(fn, args.repeat)
        if baseline is None:
            baseline = best
        speedup = baseline / best
        print(f"{name:<26}{best:>9.3f}s{median:>9.3f}s{checksum:>12}{speedup:>9.1f}x")


if __name__ == "__main__":
    main()
