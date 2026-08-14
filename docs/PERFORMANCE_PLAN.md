# PDF Assistant — Performance Overhaul Plan

A staged plan to make this app dramatically faster, modernise the stack, and fix
correctness bugs found while profiling. Each stage lands as its own commit on
`perf/fast-pipeline` and is independently reviewable.

## Baseline (as of `912ec1f`)

`PDFread.py` is a single 475-line Streamlit script:

- **Extraction** — `pdfplumber`, page-by-page, appending to a string.
- **Summarisation** — `t5-small`, one 500-char chunk at a time, 4-way beam search.
- **Q&A** — `distilbert-base-cased-distilled-squad`, run against *every* chunk of
  the document for *every* question.
- **Deps** — `streamlit==1.29.0`, `torch==2.0.1`, `transformers==4.34.0` (2023-era).

### Findings

Ordered by expected impact. "Cost" is the dominant term for an *N*-page document.

| # | Finding | Location | Impact |
|---|---|---|---|
| 1 | Q&A runs a transformer forward pass over **every chunk** — no retrieval step. A 100-page PDF is ~400 forward passes *per question*. | `answer_question` | O(N) per question |
| 2 | Models are rebuilt per Streamlit **session**, not per process. Every browser tab re-downloads and re-instantiates T5 + DistilBERT. | `PDFAssistant.__init__`, `_load_models` | Cold start per user |
| 3 | Q&A tokeniser uses `padding="max_length"` — a 40-token chunk is padded to 512 and the model does ~12x the needed work. | `_answer_question_from_context` | ~5-12x waste |
| 4 | Summarisation chunks at **500 characters** (~125 tokens) while T5's window is 512 tokens. The model is fed at ~25% capacity, so ~4x more forward passes than needed. | `generate_summary` | ~4x waste |
| 5 | `num_beams=4` on every chunk — 4x the decode cost of greedy. | `_summarize_text` | ~4x decode |
| 6 | Every chunk of the whole document is summarised, then naively concatenated. Cost grows linearly with no cap and the output is unreadable for long docs. | `generate_summary` | Unbounded |
| 7 | Extraction re-runs on every rerun; no caching keyed on file content. | `read_pdf` | Repeated work |
| 8 | `self.pdf_text += page_text` — quadratic string building on large documents. | `read_pdf` | O(N²) |
| 9 | `pdfplumber` builds a full layout/char model per page. For plain text extraction `pypdfium2` is roughly an order of magnitude faster. | `read_pdf` | Large |
| 10 | `st.progress()` fires a websocket round-trip per page/chunk. | throughout | UI stall |
| 11 | No `model.eval()`, no `inference_mode`, no quantisation, no thread tuning. | throughout | ~2-3x on CPU |

### Correctness bugs found while profiling

| # | Bug | Detail |
|---|---|---|
| A | **The chunk timeout never works.** `threading.Timer(60, timeout_handler)` raises `TimeoutException` *on the timer thread*. The `except TimeoutException` in the main thread can never catch it, so the exception is printed to stderr and the "timeout" silently does nothing — while spawning one thread per chunk. | `generate_summary`, `answer_question` |
| B | **Q&A span decode can invert.** `answer_start` and `answer_end` are `argmax`'d independently, so `end < start` is possible, yielding an empty or reversed answer. | `_answer_question_from_context` |
| C | **Confidence score is not a probability.** Raw logits are averaged and divided by an arbitrary `10.0`, then clamped — so scores are not comparable across chunks, which is exactly what the "best answer" loop relies on. | `_answer_question_from_context` |
| D | **`requirements.txt` is UTF-16LE encoded.** `pip` reads requirements as UTF-8; this file can fail to parse on a clean deploy. | `requirements.txt` |
| E | **`packages.txt` pins `python3-distutils`**, which no longer exists on current Debian images and was removed from Python itself in 3.12. | `packages.txt` |
| F | `import torch` at module scope alongside Streamlit's file watcher is a known crash source; `torch` is then redundantly re-imported inside `answer_question`. | top of file |

## Stages

### Stage 1 — Foundation: I/O, caching, and dependency hygiene

No model behaviour changes, so it is safe to land first and easy to verify.

- Replace the extraction backend with **`pypdfium2`**, keeping `pdfplumber` as an
  automatic fallback for documents pdfium reports as empty.
- Extract pages **in parallel** across a thread pool.
- Build text with a list + `join` instead of `+=`.
- Cache extraction with **`@st.cache_data`** keyed on a hash of the file bytes.
- Cache models with **`@st.cache_resource`** so they load once per process and are
  shared by every session.
- Throttle progress updates instead of writing on every item.
- Rewrite `requirements.txt` as UTF-8 on a modern, tested stack.
- Drop the obsolete `packages.txt` entry.
- Add `.gitignore`, a reproducible benchmark harness, and record baseline numbers.

**Fixes:** 2, 7, 8, 9, 10, D, E.

### Stage 2 — Inference engine

- **Batch** chunks through both models instead of looping one at a time.
- Dynamic padding (`padding=True`) instead of `padding="max_length"`.
- Token-aware chunking that actually fills the 512-token window.
- `model.eval()` + `torch.inference_mode()` + `torch.set_num_threads` tuning.
- **int8 dynamic quantisation** on CPU.
- Delete the broken `threading.Timer` machinery and replace it with a real,
  cooperative budget check.
- Correct span decoding (constrain `end >= start`, search the top-k span pairs)
  and report a real softmax probability.

**Fixes:** 3, 4, 5, 11, A, B, C.

### Stage 3 — Algorithmic: retrieval instead of brute force

- Embed chunks **once** per document and cache the matrix.
- Answer questions by retrieving the top-k chunks and running the QA model on
  those only — turning an O(N) scan into O(k).
- Map-reduce summarisation with a bounded budget and a second-pass reduction, so
  the summary stays readable and the cost stops growing linearly.

**Fixes:** 1, 6.

### Stage 4 — UX and deployment

- `st.download_button` in place of the hand-rolled base64 anchor.
- `st.fragment` so asking a question does not rerun the whole page.
- Stream results as they arrive rather than blocking behind a spinner.
- Regression tests and a CI workflow.

## Verification

Every stage is measured with `benchmarks/bench_extract.py` against generated
fixture PDFs, and the numbers are recorded in the pull request as they land.
