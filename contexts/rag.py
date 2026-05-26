"""Retrieve top-K textbook chunks for each problem by embedding cosine sim.

Corpus-agnostic: takes a list of `Chunk` objects (see contexts/corpus.py).
Leakage guard: drops any chunk containing a long verbatim substring of the
problem text.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mathreason_sae.contexts.corpus import (
    Chunk,
    chapters_match,
    load_d2_stub,
    load_jsonl_corpus,
)
from mathreason_sae.contexts.embed import embed_texts
from mathreason_sae.data.schema import ProblemRecord


LEAKAGE_THRESHOLD = 30


def _alphanum(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text.lower())


def has_leakage(chunk_text: str, problem_text: str, threshold: int = LEAKAGE_THRESHOLD) -> bool:
    c = _alphanum(chunk_text)
    p = _alphanum(problem_text)
    if len(p) < threshold:
        return False
    for start in range(len(p) - threshold + 1):
        if p[start : start + threshold] in c:
            return True
    return False


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    text: str
    author: str
    chapter: str


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return an @ bn.T


def retrieve(
    problems: list[ProblemRecord],
    chunks: list[Chunk],
    cache_dir: Path,
    top_k: int = 5,
    restrict_same_author: bool = True,
    restrict_same_chapter: bool = False,
    chapter_match_mode: str = "numeric_stem",
    chapter_fallback: str = "same_author",
) -> dict[str, list[RetrievalHit]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prob_embs = embed_texts(
        [p.question for p in problems],
        cache_dir / "problem_embeddings.json",
    )
    chunk_embs = embed_texts(
        [c.text for c in chunks],
        cache_dir / "chunk_embeddings.json",
    )
    sims = _cos(prob_embs, chunk_embs)

    results: dict[str, list[RetrievalHit]] = {}
    for i, prob in enumerate(problems):
        row = sims[i].copy()
        if restrict_same_author or restrict_same_chapter:
            author_ok = np.ones(len(chunks), dtype=bool)
            if restrict_same_author:
                author_ok = np.array([ch.author == prob.author for ch in chunks], dtype=bool)

            keep_mask = author_ok
            if restrict_same_chapter:
                chapter_ok = np.array([
                    chapters_match(prob.chapter, ch.chapter, mode=chapter_match_mode)
                    for ch in chunks
                ], dtype=bool)
                chapter_mask = keep_mask & chapter_ok
                if chapter_mask.any():
                    keep_mask = chapter_mask
                elif chapter_fallback == "same_author" and restrict_same_author:
                    keep_mask = author_ok
                elif chapter_fallback == "none":
                    keep_mask = chapter_mask
                else:
                    raise ValueError(f"unknown chapter fallback mode: {chapter_fallback}")

            row[~keep_mask] = -1.0

        order = np.argsort(-row)
        hits: list[RetrievalHit] = []
        for j in order:
            if row[j] < 0:
                break
            ch = chunks[int(j)]
            if has_leakage(ch.text, prob.question):
                continue
            hits.append(RetrievalHit(
                chunk_id=ch.chunk_id, score=float(row[j]),
                text=ch.text, author=ch.author, chapter=str(ch.chapter),
            ))
            if len(hits) >= top_k:
                break
        results[prob.problem_id] = hits
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True, help="JSONL corpus or d2.csv stub")
    ap.add_argument("--corpus-format", choices=["jsonl", "d2_stub"], default="jsonl")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-restrict-author", action="store_true")
    ap.add_argument("--restrict-chapter", action="store_true")
    ap.add_argument("--chapter-match-mode", choices=["prefix", "numeric_stem"],
                    default="numeric_stem")
    ap.add_argument("--chapter-fallback", choices=["same_author", "none"],
                    default="same_author")
    args = ap.parse_args()

    problems = []
    with open(args.problems) as f:
        for line in f:
            problems.append(ProblemRecord.from_json(line))

    if args.corpus_format == "jsonl":
        chunks = load_jsonl_corpus(args.corpus)
    else:
        chunks = load_d2_stub(args.corpus)
    print(f"loaded {len(problems)} problems, {len(chunks)} chunks")

    hits = retrieve(
        problems, chunks, cache_dir=args.cache_dir,
        top_k=args.top_k,
        restrict_same_author=not args.no_restrict_author,
        restrict_same_chapter=args.restrict_chapter,
        chapter_match_mode=args.chapter_match_mode,
        chapter_fallback=args.chapter_fallback,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(args.out, "w") as f:
        for pid, hl in hits.items():
            f.write(json.dumps({
                "problem_id": pid,
                "hits": [h.__dict__ for h in hl],
            }) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
