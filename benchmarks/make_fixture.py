"""Generate a reproducible, text-heavy PDF fixture for the extraction benchmark.

Usage:
    python benchmarks/make_fixture.py --pages 60 --out benchmarks/fixture_60p.pdf
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

WORDS = (
    "document retrieval transformer summarisation inference latency throughput "
    "tokenizer embedding attention pipeline extraction corpus heuristic gradient "
    "quantisation checkpoint benchmark deployment container concurrency cache "
    "vector index similarity relevance passage context window batch schedule"
).split()


def _paragraph(rng: random.Random, sentences: int = 6) -> str:
    out = []
    for _ in range(sentences):
        n = rng.randint(12, 26)
        words = rng.choices(WORDS, k=n)
        out.append(words[0].capitalize() + " " + " ".join(words[1:]) + ".")
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/fixture_60p.pdf"))
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    styles = getSampleStyleSheet()

    # Roughly 5 paragraphs per page at this style, so scale the flowable count
    # to the requested page count and let the layout engine paginate.
    story = []
    for i in range(args.pages * 5):
        if i % 5 == 0:
            story.append(Paragraph(f"Section {i // 5 + 1}", styles["Heading2"]))
        story.append(Paragraph(_paragraph(rng), styles["BodyText"]))
        story.append(Spacer(1, 6))

    SimpleDocTemplate(str(args.out), pagesize=LETTER).build(story)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
