# PDF Assistant

A Streamlit app that reads a PDF, summarises it, and answers questions about it.
Everything runs locally on CPU — no API keys, no data leaves the machine.

- **Extraction** — pypdfium2 (the engine Chrome uses), with pdfplumber as a fallback
- **Summarisation** — `t5-small`, map-reduce over the document
- **Q&A** — BM25 retrieval to find the relevant sections, then
  `distilbert-base-cased-distilled-squad` reads only those

## Quick start

```bash
pip install -r requirements.txt
streamlit run PDFread.py
```

Then upload a PDF in the sidebar and press **Process PDF**.

The models (~500 MB) download from Hugging Face on first use and are cached
afterwards. The first request to a fresh server waits ~18s for them to load and
quantise; that cost is paid once per server process, not per user or question.

> **macOS:** `requirements.txt` pins `torch==2.13.0+cpu` from PyTorch's own
> index, which has no macOS wheels. Install plain `torch==2.13.0` from PyPI
> instead — it is already CPU-only there.

## How it works

```
PDF ──► pypdfium2 ──► token-aware chunks ──► BM25 index
                                │                │
                                │                └──► top 5 chunks ──► DistilBERT ──► answer
                                │
                                └──► 24 sampled chunks ──► T5 ──► reduce ──► summary
```

Three decisions do most of the work:

- **Chunks are sized in tokens, not characters**, so they fill the models'
  512-token window instead of a quarter of it.
- **Q&A retrieves before it reads.** Scoring spans across every chunk gave the
  model a fresh chance to be confidently wrong on each one; reading five
  instead of sixty-five raised the share of answers drawn from the correct
  chunk from 25% to 60%.
- **Summarisation is capped and then reduced.** Summarising every chunk and
  concatenating produced output as long as the document. Cost and output length
  no longer track document length.

## Performance

Against the original implementation, on CPU:

| | Before | After | |
|---|---|---|---|
| Text extraction | | | **~60x** |
| Q&A, per question | 8.11s | **0.64s** | **12.7x** |
| Summarisation (20k chars) | 61.6s | **5.6s** | **11.0x** |

Every figure is reproducible — see [`benchmarks/README.md`](benchmarks/README.md),
which also records what was tried and **rejected** (parallel extraction, static
embeddings, embedding-based chunk selection) and why.

## Development

```bash
pip install -r requirements-dev.txt
pytest              # 52 tests, ~1s
mypy                # config in pyproject.toml
```

The test suite deliberately excludes torch and transformers: a stubbed
tokenizer stands in, so chunking, retrieval, the time budget and document
loading are all covered in about a second. Behaviour that genuinely needs a
model is measured by the scripts in `benchmarks/` instead.

```bash
python benchmarks/make_fixture.py --pages 60 --out benchmarks/fixture_60p.pdf
python benchmarks/bench_extract.py   benchmarks/fixture_60p.pdf
python benchmarks/bench_inference.py benchmarks/fixture_60p.pdf --chars 20000
python benchmarks/bench_retrieval.py benchmarks/fixture_60p.pdf --questions 20
```

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `PDFREAD_QUANTIZE` | `1` | int8 dynamic quantisation, worth ~1.24x. Set to `0` to disable. |

Tunables live at the top of their modules: `TOP_K` in `retrieval.py`,
`SUMMARY_MAX_MAP_CHUNKS` and the token budgets in `inference.py`, and the
wall-clock budgets in `PDFread.py`.

## Limitations

- **Scanned PDFs are not supported.** Neither backend does OCR; a document with
  no text layer reports that rather than pretending to work.
- **`t5-small` confabulates** on long-form prose, sometimes attributing
  statements to invented names. That is the model, not the pipeline — a
  stronger summariser is the fix, not more chunking.
- **Answers are extractive.** Q&A returns a span copied from the document, so
  it cannot synthesise an answer spread across several sections.
