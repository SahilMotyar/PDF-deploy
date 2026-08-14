# Benchmarks

## Running

```bash
python benchmarks/make_fixture.py --pages 60 --out benchmarks/fixture_60p.pdf
python benchmarks/bench_extract.py benchmarks/fixture_60p.pdf --repeat 3
```

The fixture generator is seeded, so the document is byte-identical across runs.

## Extraction backends

Windows 11, Python 3.12, best of 3 runs.

| Document | pdfplumber | pypdfium2 | Speedup |
|---|---|---|---|
| 76 pages, 140 KiB | 8.138s | **0.130s** | **62.5x** |
| 202 pages, 369 KiB | 22.016s | **0.365s** | **60.3x** |

pdfium extracts about 1% more characters than pdfplumber (331,663 vs 328,150 on
the 76-page fixture), from differences in whitespace and line-break handling.

## Why extraction is not parallelised

pdfium is not thread-safe and pypdfium2 adds no locking, so any parallel version
has to be process-based. That was implemented and measured, and it lost:

| Document | pypdfium2 sequential | pypdfium2, 4 processes |
|---|---|---|
| 76 pages | **0.130s** | 0.395s |
| 202 pages | **0.365s** | 0.515s |

The sequential path already reads roughly 550 pages/second, so pool startup and
the cost of shipping the document to each worker dominate. Amortising that would
take a document of several thousand pages, and even then the win would be small
next to the cost of an extra code path and the per-worker memory. The parallel
implementation was removed rather than shipped unused.
