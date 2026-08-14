"""A/B the original inference path against the Stage 2 one.

Downloads t5-small and distilbert-base-cased-distilled-squad on first run.

Usage:
    python benchmarks/bench_inference.py benchmarks/fixture_60p.pdf --chars 20000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference  # noqa: E402
import pdf_extract  # noqa: E402


# --- The original implementation, reproduced for comparison ------------------


def old_split_text(text, max_length=1000, overlap=100):
    """Character-based chunking, as originally written."""
    sentences = inference.split_sentences(text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_length:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            point = max(0, len(current) - overlap)
            current = current[point:] + " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


def old_summarize(chunks, tokenizer, model):
    import torch

    out = []
    for chunk in chunks:
        if len(chunk) < 100:
            continue
        inputs = tokenizer(
            "summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True
        )
        with torch.no_grad():
            ids = model.generate(
                inputs.input_ids,
                max_length=100,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True,
            )
        out.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return out


def old_answer(question, chunks, tokenizer, model):
    import torch

    best_answer, highest = "", 0.0
    for chunk in chunks:
        inputs = tokenizer(
            question,
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        with torch.no_grad():
            outputs = model(**inputs)
            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits) + 1
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs.input_ids[0][start:end])
            )
        confidence = (
            float(torch.max(outputs.start_logits).item() + torch.max(outputs.end_logits).item()) / 2
        )
        score = min(1.0, max(0.0, confidence / 10.0))
        if score > highest and answer.strip():
            highest, best_answer = score, answer
    return best_answer, highest


# --- Harness -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--chars", type=int, default=20000, help="truncate the document")
    ap.add_argument("--question", default="What does the document describe?")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    inference.split_sentences("warm up the tokenizer download.")

    text = pdf_extract.extract(args.pdf.read_bytes()).text[: args.chars]
    print(f"document: {len(text)} chars\n")

    from transformers import (
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    print("loading models...")
    sum_tok = AutoTokenizer.from_pretrained(inference.SUMMARIZER_MODEL)
    qa_tok = AutoTokenizer.from_pretrained(inference.QA_MODEL)
    sum_model_raw = AutoModelForSeq2SeqLM.from_pretrained(inference.SUMMARIZER_MODEL)
    qa_model_raw = AutoModelForQuestionAnswering.from_pretrained(inference.QA_MODEL)

    quantize = not args.no_quantize
    sum_model_new = inference.prepare(
        AutoModelForSeq2SeqLM.from_pretrained(inference.SUMMARIZER_MODEL), quantize
    )
    qa_model_new = inference.prepare(
        AutoModelForQuestionAnswering.from_pretrained(inference.QA_MODEL), quantize
    )
    print(f"quantized: {quantize}\n")

    # Chunking
    old_sum_chunks = old_split_text(text, max_length=500, overlap=50)
    old_qa_chunks = old_split_text(text, max_length=1000, overlap=100)
    new_sum_chunks = inference.chunk_by_tokens(text, sum_tok, inference.SUMMARY_INPUT_TOKENS)
    new_qa_chunks = inference.chunk_by_tokens(text, qa_tok, inference.QA_INPUT_TOKENS)

    print("chunk counts (fewer chunks = fewer forward passes)")
    print(f"  summarise: {len(old_sum_chunks)} -> {len(new_sum_chunks)}")
    print(f"  qa:        {len(old_qa_chunks)} -> {len(new_qa_chunks)}\n")

    # Q&A
    print(f"Q&A: {args.question!r}")
    t0 = time.perf_counter()
    old_ans, old_score = old_answer(args.question, old_qa_chunks, qa_tok, qa_model_raw)
    old_qa_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    new = inference.answer_from_chunks(args.question, new_qa_chunks, qa_tok, qa_model_new)
    new_qa_time = time.perf_counter() - t0

    print(f"  before: {old_qa_time:7.2f}s  score={old_score:.3f}  {old_ans[:70]!r}")
    print(f"  after:  {new_qa_time:7.2f}s  score={new.score if new else 0:.3f}  "
          f"{(new.text[:70] if new else '')!r}")
    print(f"  speedup: {old_qa_time / new_qa_time:.1f}x\n")

    # Summarisation
    print("Summarisation")
    t0 = time.perf_counter()
    old_summaries = old_summarize(old_sum_chunks, sum_tok, sum_model_raw)
    old_sum_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_summaries = inference.summarize_chunks(new_sum_chunks, sum_tok, sum_model_new)
    new_sum_time = time.perf_counter() - t0

    print(f"  before: {old_sum_time:7.2f}s  {len(old_summaries)} summaries, "
          f"{sum(len(s) for s in old_summaries)} chars")
    print(f"  after:  {new_sum_time:7.2f}s  {len(new_summaries)} summaries, "
          f"{sum(len(s) for s in new_summaries)} chars")
    print(f"  speedup: {old_sum_time / new_sum_time:.1f}x\n")

    total_old, total_new = old_qa_time + old_sum_time, new_qa_time + new_sum_time
    print(f"total: {total_old:.2f}s -> {total_new:.2f}s  ({total_old / total_new:.1f}x)")


if __name__ == "__main__":
    main()
