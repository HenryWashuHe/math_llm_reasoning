"""Relevant-looking-wrong sham context builder.

Sham = chunks from the same textbook, in the same broad domain (linear
algebra), that are *topically disjoint* from the problem. We want context
that looks like it could help — same author's voice, same math vocabulary,
theorem-style prose — but addresses a different subroutine.

Strategy:
  1. Bucket chunks by task-family-style topic using keyword matching on the
     taxonomy. Chunks that match zero family keywords go into "general".
  2. For each problem, take its task_family label and pick chunks from
     *different* families (prefer same author, then cross-author).
  3. Leakage guard: reject any chunk containing long verbatim substrings of
     the problem text.
  4. Length-match: top_k chunks whose combined character length lands within
     ±15% of the real retrieved context's length, so length isn't a
     confound.

If a problem has no task_family (unlabeled), falls back to chapter-disjoint
chunks from the same author.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from mathreason_sae.contexts.corpus import (
    Chunk,
    chapters_match,
    load_d2_stub,
    load_jsonl_corpus,
)
from mathreason_sae.contexts.rag import has_leakage
from mathreason_sae.data.schema import ProblemRecord


def _match_families(chunk_text: str, families: list[dict]) -> set[str]:
    matched = set()
    t = chunk_text.lower()
    for fam in families:
        for kw in fam.get("keywords", []):
            if kw.lower() in t:
                matched.add(fam["id"])
                break
        if fam["id"] in matched:
            continue
        for pat in fam.get("patterns", []):
            if re.search(pat, t):
                matched.add(fam["id"])
                break
    return matched


def build_topic_buckets(chunks: list[Chunk], taxonomy: dict) -> dict[str, set[int]]:
    """family_id -> set of chunk indices whose text mentions that family."""
    families = taxonomy["families"]
    buckets: dict[str, set[int]] = {fam["id"]: set() for fam in families}
    for i, ch in enumerate(chunks):
        fams = _match_families(ch.text, families)
        for f in fams:
            buckets[f].add(i)
    return buckets


@dataclass
class ShamChoice:
    problem_id: str
    chunk_ids: list[str]
    texts: list[str]
    target_length: int
    actual_length: int


def build_sham(
    problem: ProblemRecord,
    retrieved_hits_len: int,
    chunks: list[Chunk],
    topic_buckets: dict[str, set[int]],
    top_k: int,
    rng: random.Random,
    length_tol: float = 0.15,
    exclude_same_chapter: bool = False,
    chapter_match_mode: str = "numeric_stem",
) -> Optional[ShamChoice]:
    candidate_idx: list[int] = []

    if problem.task_family:
        own = topic_buckets.get(problem.task_family, set())
        other_family_idxs: set[int] = set()
        for fam, s in topic_buckets.items():
            if fam == problem.task_family:
                continue
            other_family_idxs |= s
        candidate_idx = list(other_family_idxs - own)
    else:
        candidate_idx = [i for i, c in enumerate(chunks)
                         if not chapters_match(problem.chapter, c.chapter, mode=chapter_match_mode)]

    if exclude_same_chapter:
        candidate_idx = [
            i for i in candidate_idx
            if not chapters_match(problem.chapter, chunks[i].chapter, mode=chapter_match_mode)
        ]

    same_author = [i for i in candidate_idx if chunks[i].author == problem.author]
    diff_author = [i for i in candidate_idx if chunks[i].author != problem.author]
    rng.shuffle(same_author)
    rng.shuffle(diff_author)
    ordered = same_author + diff_author

    clean = [i for i in ordered if not has_leakage(chunks[i].text, problem.question)]
    if len(clean) < top_k:
        return None

    low = int(retrieved_hits_len * (1 - length_tol))
    high = int(retrieved_hits_len * (1 + length_tol))

    best: Optional[list[int]] = None
    best_diff = None
    for _ in range(50):
        rng.shuffle(clean)
        pick = clean[:top_k]
        total = sum(len(chunks[i].text) for i in pick)
        if low <= total <= high:
            best = pick
            break
        diff = abs(total - retrieved_hits_len)
        if best_diff is None or diff < best_diff:
            best = pick
            best_diff = diff
    if best is None:
        return None

    return ShamChoice(
        problem_id=problem.problem_id,
        chunk_ids=[chunks[i].chunk_id for i in best],
        texts=[chunks[i].text for i in best],
        target_length=retrieved_hits_len,
        actual_length=sum(len(chunks[i].text) for i in best),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=Path, required=True,
                    help="labeled+split JSONL")
    ap.add_argument("--retrieved", type=Path, required=True,
                    help="retrieval JSONL from rag.py")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--corpus-format", choices=["jsonl", "d2_stub"], default="jsonl")
    ap.add_argument("--taxonomy", type=Path,
                    default=Path(__file__).resolve().parent.parent / "configs" / "task_families.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-same-chapter", action="store_true")
    ap.add_argument("--chapter-match-mode", choices=["prefix", "numeric_stem"],
                    default="numeric_stem")
    args = ap.parse_args()

    with open(args.taxonomy) as f:
        taxonomy = yaml.safe_load(f)

    problems = []
    with open(args.problems) as f:
        for line in f:
            problems.append(ProblemRecord.from_json(line))

    retrieved_len: dict[str, int] = {}
    with open(args.retrieved) as f:
        for line in f:
            r = json.loads(line)
            retrieved_len[r["problem_id"]] = sum(len(h["text"]) for h in r["hits"])

    chunks = (load_jsonl_corpus(args.corpus) if args.corpus_format == "jsonl"
              else load_d2_stub(args.corpus))
    buckets = build_topic_buckets(chunks, taxonomy)
    print(f"loaded {len(problems)} problems, {len(chunks)} chunks")
    print(f"buckets: {', '.join(f'{k}={len(v)}' for k,v in buckets.items())}")

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_skip = 0, 0
    with open(args.out, "w") as f:
        for prob in problems:
            ret_len = retrieved_len.get(prob.problem_id, 0)
            if ret_len == 0:
                n_skip += 1
                continue
            sc = build_sham(
                prob,
                ret_len,
                chunks,
                buckets,
                args.top_k,
                rng,
                exclude_same_chapter=args.exclude_same_chapter,
                chapter_match_mode=args.chapter_match_mode,
            )
            if sc is None:
                n_skip += 1
                continue
            f.write(json.dumps(sc.__dict__) + "\n")
            n_ok += 1
    print(f"wrote {n_ok} sham records, skipped {n_skip} -> {args.out}")


if __name__ == "__main__":
    main()
