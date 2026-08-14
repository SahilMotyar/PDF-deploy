"""Batched, CPU-tuned inference for summarisation and extractive QA.

Kept free of Streamlit imports so it can be benchmarked and tested directly,
the same way pdf_extract.py is.

Three things dominate the cost of the original implementation, and all three
are addressed here:

* Chunks were built to 500 *characters* (~125 tokens) against a 512-token
  window, so the model ran roughly four times more often than necessary.
  Chunking is now token-aware and fills the window.
* Every chunk was pushed through the model on its own, one forward pass at a
  time. Chunks are now batched.
* The QA tokenizer used ``padding="max_length"``, padding a 40-token chunk out
  to 512. Padding is now dynamic, to the longest member of each batch.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Sequence

SUMMARIZER_MODEL = "t5-small"
QA_MODEL = "distilbert-base-cased-distilled-squad"

# Both models take 512 tokens. Leave headroom for special tokens and, for T5,
# the "summarize: " task prefix. The QA budget is smaller because the question
# shares the window with the context; overshooting would silently truncate the
# tail of every chunk.
SUMMARY_INPUT_TOKENS = 480
QA_INPUT_TOKENS = 440

# Sentences of overlap context carried between adjacent chunks.
OVERLAP_TOKENS = 48

# An extractive answer longer than this is almost always a decoding artefact.
MAX_ANSWER_TOKENS = 40

SUMMARY_MAX_NEW_TOKENS = 100
SUMMARY_MIN_NEW_TOKENS = 30

# Beam search multiplies decode cost by the beam count. Four beams was the
# original setting; two keeps most of the quality for half the work.
SUMMARY_NUM_BEAMS = 2

ProgressFn = Callable[[int, int], None]


def _emit(progress: ProgressFn | None, done: int, total: int) -> None:
    if progress is not None:
        progress(done, total)


class Budget:
    """A cooperative wall-clock budget, checked between batches.

    The original code used ``threading.Timer`` to raise ``TimeoutException``.
    That raises on the *timer* thread, where the caller's ``except`` clause can
    never catch it, so the timeout never worked and every chunk leaked a
    thread. Checking a deadline between batches is both correct and free.

    Granularity is one batch: an in-flight batch always finishes, so the wall
    clock can overshoot the budget by roughly one batch's duration.
    """

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def expired(self) -> bool:
        return self.seconds is not None and self.elapsed >= self.seconds


@dataclass(frozen=True)
class Answer:
    text: str
    score: float
    chunk_index: int


# --- Model preparation -------------------------------------------------------


def quantization_enabled() -> bool:
    """int8 dynamic quantisation, on unless PDFREAD_QUANTIZE=0."""
    return os.environ.get("PDFREAD_QUANTIZE", "1") != "0"


def prepare(model, quantize: bool | None = None):
    """Put a model into inference shape.

    ``eval()`` disables dropout, which the original code never turned off.

    Dynamic int8 quantisation is worth about 1.24x on top of batching (7.93s
    to 6.38s on the benchmark). Note that torch 2.13 emits a DeprecationWarning
    for the quantized tensor constructors this path uses, and PyTorch intends
    to remove them (pytorch/pytorch#184982). It still works today, it is worth
    the gain, and ``PDFREAD_QUANTIZE=0`` turns it off; when it is finally
    removed the replacement is torchao. Failure to quantise is never fatal.
    """
    import torch

    model.eval()

    if quantize is None:
        quantize = quantization_enabled()
    if quantize:
        try:
            from torch.ao.quantization import quantize_dynamic

            model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception:
            # Quantisation is an optimisation, never a requirement.
            pass
    return model


# --- Chunking ----------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Sentence-split with NLTK, falling back to a naive split."""
    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        return [s.strip() + "." for s in text.split(".") if s.strip()]


def chunk_by_tokens(
    text: str,
    tokenizer,
    budget_tokens: int = SUMMARY_INPUT_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split text into chunks that fill the model's token window.

    Sentence boundaries are respected so chunks do not start mid-clause, but
    the packing target is tokens rather than characters, which is what the
    model actually limits.
    """
    if not text or text.isspace():
        return []

    sentences = [s for s in (s.strip() for s in split_sentences(text)) if s]
    if not sentences:
        return []

    # One batched call; the Rust tokenizer makes this far cheaper than
    # measuring each sentence separately.
    encoded = tokenizer(sentences, add_special_tokens=False)["input_ids"]

    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append(" ".join(s for s, _ in current))

    for sentence, ids in zip(sentences, encoded):
        length = len(ids)

        # A single sentence longer than the window has to be hard-split.
        if length > budget_tokens:
            flush()
            current, current_tokens = [], 0
            for start in range(0, length, budget_tokens):
                piece = tokenizer.decode(
                    ids[start : start + budget_tokens], skip_special_tokens=True
                ).strip()
                if piece:
                    chunks.append(piece)
            continue

        if current_tokens + length > budget_tokens and current:
            flush()
            # Carry a tail of whole sentences as overlap, so a fact spanning a
            # chunk boundary is still visible in one piece.
            kept: list[tuple[str, int]] = []
            kept_tokens = 0
            for sent, sent_len in reversed(current):
                if kept_tokens + sent_len > overlap_tokens:
                    break
                kept.insert(0, (sent, sent_len))
                kept_tokens += sent_len
            current, current_tokens = kept, kept_tokens

        current.append((sentence, length))
        current_tokens += length

    flush()
    return chunks


def _batches(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


# --- Summarisation -----------------------------------------------------------


def summarize_chunks(
    chunks: Sequence[str],
    tokenizer,
    model,
    batch_size: int = 4,
    num_beams: int = SUMMARY_NUM_BEAMS,
    budget: Budget | None = None,
    progress: ProgressFn | None = None,
) -> list[str]:
    """Summarise chunks in batches, returning one summary per chunk."""
    import torch

    if not chunks:
        return []

    summaries: list[str] = []
    total = len(chunks)

    for start, batch in _batches(chunks, batch_size):
        if budget is not None and budget.expired:
            break

        encoded = tokenizer(
            ["summarize: " + chunk for chunk in batch],
            return_tensors="pt",
            padding=True,          # to the longest in this batch, not to 512
            truncation=True,
            max_length=512,
        )

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
                min_new_tokens=SUMMARY_MIN_NEW_TOKENS,
                length_penalty=2.0,
                num_beams=num_beams,
                early_stopping=True,
            )

        summaries.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        _emit(progress, min(start + len(batch), total), total)

    return summaries


# --- Extractive QA -----------------------------------------------------------


def _best_span(start_logits, end_logits, valid, max_answer_tokens: int, top_k: int = 20):
    """Pick the highest-probability valid span.

    Fixes two bugs in the original: `start` and `end` were argmax'd
    independently, so `end < start` was reachable and produced an empty or
    reversed answer; and the "confidence" was an average of two raw logits
    divided by an arbitrary constant, which is not comparable across chunks --
    exactly what the best-answer loop relied on. This returns the product of
    two softmax probabilities, which is a real value in [0, 1].
    """
    import torch

    floor = torch.finfo(start_logits.dtype).min
    start_probs = torch.softmax(start_logits.masked_fill(~valid, floor), dim=-1)
    end_probs = torch.softmax(end_logits.masked_fill(~valid, floor), dim=-1)

    k = min(top_k, int(valid.sum().item()) or 1)
    top_starts = torch.topk(start_probs, k)
    top_ends = torch.topk(end_probs, k)

    best_score, best_span = 0.0, None
    for s_idx, s_prob in zip(top_starts.indices.tolist(), top_starts.values.tolist()):
        for e_idx, e_prob in zip(top_ends.indices.tolist(), top_ends.values.tolist()):
            if e_idx < s_idx or e_idx - s_idx + 1 > max_answer_tokens:
                continue
            score = s_prob * e_prob
            if score > best_score:
                best_score, best_span = score, (s_idx, e_idx)

    return best_score, best_span


def answer_from_chunks(
    question: str,
    chunks: Sequence[str],
    tokenizer,
    model,
    batch_size: int = 8,
    budget: Budget | None = None,
    progress: ProgressFn | None = None,
) -> Answer | None:
    """Answer `question` from the best-scoring span across `chunks`."""
    import torch

    if not chunks or not question.strip():
        return None

    best: Answer | None = None
    total = len(chunks)

    for start, batch in _batches(chunks, batch_size):
        if budget is not None and budget.expired:
            break

        encoded = tokenizer(
            [question] * len(batch),
            list(batch),
            return_tensors="pt",
            padding=True,               # dynamic, not padding="max_length"
            truncation="only_second",   # never truncate the question itself
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")

        with torch.inference_mode():
            outputs = model(**encoded)

        for row, chunk in enumerate(batch):
            # sequence_ids marks question tokens 0, context tokens 1, and
            # specials/padding None, so this confines answers to the context.
            sequence_ids = encoded.sequence_ids(row)
            valid = torch.tensor(
                [sid == 1 for sid in sequence_ids], dtype=torch.bool
            )
            if not bool(valid.any()):
                continue

            score, span = _best_span(
                outputs.start_logits[row],
                outputs.end_logits[row],
                valid,
                MAX_ANSWER_TOKENS,
            )
            if span is None:
                continue

            # Slice the original text by character offsets rather than
            # stitching word pieces back together, which mangles subwords.
            char_start = int(offsets[row][span[0]][0])
            char_end = int(offsets[row][span[1]][1])
            text = chunk[char_start:char_end].strip()

            if text and (best is None or score > best.score):
                best = Answer(text=text, score=score, chunk_index=start + row)

        _emit(progress, min(start + len(batch), total), total)

    return best
