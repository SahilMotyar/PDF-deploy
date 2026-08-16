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

## Retrieval (Stage 3)

Measured on 121,723 characters of real expository prose (Darwin, *On the Origin
of Species*, public domain) rather than the generated fixture, because the
fixture is random vocabulary and tells you nothing about whether retrieval
retrieves the right thing. 65 chunks, 20 queries.

Queries are sentences lifted from a known chunk, so the source chunk is known
without hand labelling. "Partial wording" keeps a shuffled 40% of the longer
words, standing in for someone phrasing a question in their own words.

| | Result |
|---|---|
| recall@5, verbatim wording | **100%** |
| recall@5, partial wording | **100%** |
| index build | 15ms |

| Per question | Full scan | Top-5 | |
|---|---|---|---|
| Time | 8.107s | **0.639s** | **12.7x** |
| Answer drawn from the correct chunk | 5/20 (25%) | **12/20 (60%)** | |

### Reading less makes the answers better

Retrieval was meant to buy speed. It also more than doubled the share of
answers taken from the correct chunk, and that is not a coincidence: scoring
spans across all 65 chunks gives the QA model 65 chances to return a confident
span from text that has nothing to do with the question, and "highest score
wins" then selects one. Reading 5 chunks removes 60 opportunities to be wrong.

This is also why answers match the exhaustive scan only 20% of the time. Low
agreement is the point -- the exhaustive scan is the thing being corrected, so
agreement with it is not a quality target.

### BM25 beat static embeddings

Static embeddings (model2vec `potion-base-8M`) were implemented, benchmarked
against BM25 and in a reciprocal-rank-fusion hybrid, and then removed:

| recall@5 | Verbatim | Partial wording |
|---|---|---|
| **BM25** | **100%** | **100%** |
| Static embeddings | 90% | 65% |
| Hybrid (RRF) | 95% | 85% |

The dense signal did not merely fail to help, it pulled the hybrid below plain
BM25 on both. Removing it also drops a dependency and a 30 MB model download.

**Caveat, stated plainly:** both query sets are derived from the document's own
text, which structurally favours lexical matching. A genuine paraphrase
benchmark -- human-written questions using different vocabulary -- is the case
where embeddings would be expected to earn their place, and it was not run.
The conclusion supported by this evidence is "BM25 is better here", not
"embeddings are useless".

## Map-reduce summarisation (Stage 3)

Same document, 65 summarisable chunks.

| | Chunks read | Time | Output | Share of document |
|---|---|---|---|---|
| Stage 2 (every chunk, joined) | 65 | 92.7s | 16,741 chars | 13.8% |
| Stage 3 (capped, then reduced) | **24** | **43.1s** | **899 chars** | **0.7%** |

2.2x faster, and the output is 19x shorter. Length is the more important half:
concatenating a summary of every chunk produces something that grows with the
document and is not a summary in any useful sense. Capping the map phase at 24
evenly spaced chunks and folding the results means cost and output length stop
tracking document length at all.

Reduction stops at roughly 2,000 characters rather than folding all the way
down. Folding to a single pass gave one 284-character blurb for a 121,000
-character book, which is too terse to be worth reading.

### Known limitation

`t5-small` confabulates on this material -- the generated summary attributes
statements to invented names. That is the model, not the pipeline, and no
amount of chunking fixes it. Choosing a stronger summarisation model is a
separate change from making the existing one fast.
