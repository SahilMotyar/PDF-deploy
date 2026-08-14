# Benchmarks

## Running

```bash
python benchmarks/make_fixture.py --pages 60 --out benchmarks/fixture_60p.pdf
python benchmarks/bench_extract.py benchmarks/fixture_60p.pdf --repeat 3
python benchmarks/bench_inference.py benchmarks/fixture_60p.pdf --chars 20000
```

The fixture generator is seeded, so the document is byte-identical across runs.
`bench_inference.py` downloads t5-small and DistilBERT on first run, and
reproduces the original inference path in-file so the comparison is like for
like.

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

## Inference (Stage 2)

20,000 characters of the fixture, CPU, one run.

| Task | Before | After | Speedup |
|---|---|---|---|
| Q&A (one question) | 4.50s | **0.79s** | **5.7x** |
| Summarisation | 61.59s | **5.58s** | **11.0x** |
| Combined | 66.10s | **6.38s** | **10.4x** |

Most of it comes from doing less work rather than doing it faster. Token-aware
chunking fills the 512-token window instead of a quarter of it, which collapses
the number of forward passes outright:

| Chunks over the same text | Before | After |
|---|---|---|
| Summarisation | 56 | **8** |
| Q&A | 25 | **8** |

The rest is batching, dynamic padding in place of `padding="max_length"`, and
dropping beam search from 4 to 2.

### int8 quantisation

Dynamic quantisation is worth about 1.24x *on top of* the above:

| | Total |
|---|---|
| Batching only | 7.93s |
| Batching + int8 | **6.38s** |

It is on by default and `PDFREAD_QUANTIZE=0` disables it. Caveat: torch 2.13
deprecates the quantized tensor constructors this uses and intends to remove
them ([pytorch#184982](https://github.com/pytorch/pytorch/issues/184982)); the
successor is torchao.

## QA decoding correctness

The original decoder took `argmax` of the start and end logits independently.
Measured against five unrelated chunks for one question -- which is what most of
a document looks like relative to any single question:

| | Original | Now |
|---|---|---|
| Spans with `end < start` (empty answer) | **2 / 5** | 0 / 5 |
| Scores pinned to exactly 0.0 or 1.0 | **5 / 5** | 0 / 5 |

The inverted spans return an empty string. The pinned scores matter because the
"best answer across chunks" loop ranks on that number: once a chunk saturates at
1.0, the strict `>` comparison means no later chunk can ever displace it.

On clean prose where the model is confident, both decoders answer identically
(5/5 correct on a factual passage), so this is a robustness fix rather than an
accuracy win.
