"""Corpus adapter: load textbook chunks from any source into a common schema.

The final RAG corpus is being re-scraped and re-chunked by a teammate. To
avoid being coupled to today's `d2.csv` format, every downstream module
consumes `Chunk` records, not raw rows. New corpora implement a loader that
yields `Chunk`s and we are done.

Expected chunk fields:
  chunk_id    stable unique id across the whole corpus
  text        the passage (LaTeX preserved)
  author      textbook author slug (axler, strang, ...)
  chapter     chapter label (e.g. "8A", "3", "VII")
  section     optional sub-chapter label
  source_id   optional reference (paragraph/theorem number) for citations
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
import csv
import json
import re
import sys


@dataclass
class Chunk:
    chunk_id: str
    text: str
    author: str
    chapter: str
    section: Optional[str] = None
    source_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def load_jsonl_corpus(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(
                chunk_id=str(d["chunk_id"]),
                text=d["text"],
                author=d["author"],
                chapter=str(d.get("chapter", "")),
                section=d.get("section"),
                source_id=d.get("source_id"),
            ))
    return chunks


def load_d2_v2_csv(path: Path) -> list[Chunk]:
    """Loader for the rescraped d2 corpus (paragraph-level, 3-column schema).

    Schema: author, chapter, text. One row -> one Chunk. No grouping needed.
    LaTeX braces are intact in this file (unlike the sentence-level d2.csv).
    """
    csv.field_size_limit(sys.maxsize)
    chunks: list[Chunk] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            author = (row.get("author") or "").strip().lower()
            chapter = str(row.get("chapter") or "").strip()
            text = (row.get("text") or "").strip()
            if not text or not author:
                continue
            cid = f"{author}-{chapter}-{i:05d}"
            chunks.append(Chunk(chunk_id=cid, text=text, author=author,
                                chapter=chapter))
    return chunks


def load_d2_stub(path: Path, group_sentences: int = 8) -> list[Chunk]:
    """Stand-in loader for the current d2.csv sentence file.

    d2.csv is one-sentence-per-row. This groups consecutive sentences from the
    same (author, chapter) into rough chunks so downstream code has something
    to run against while the real corpus is being produced. Delete once the
    teammate's chunker lands.
    """
    csv.field_size_limit(sys.maxsize)
    current: list[str] = []
    cur_author: Optional[str] = None
    cur_chapter: Optional[str] = None
    chunks: list[Chunk] = []

    def flush() -> None:
        nonlocal current, cur_author, cur_chapter
        if not current or cur_author is None:
            current = []
            return
        text = " ".join(current).strip()
        cid = f"{cur_author}-{cur_chapter}-{len(chunks):06d}"
        chunks.append(Chunk(chunk_id=cid, text=text, author=cur_author,
                            chapter=str(cur_chapter)))
        current = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            author = (row.get("author") or "").strip().lower()
            chapter = str(row.get("chapter") or "").strip()
            sent = (row.get("sentence") or "").strip()
            if not sent or not author:
                continue
            if author != cur_author or chapter != cur_chapter:
                flush()
                cur_author, cur_chapter = author, chapter
            current.append(sent)
            if len(current) >= group_sentences:
                flush()
    flush()
    return chunks


def chapter_numeric_stem(chapter: str | None) -> str | None:
    raw = str(chapter or "").strip().lower()
    if not raw:
        return None
    m = re.search(r"\d+", raw)
    if m is None:
        return None
    return m.group(0)


def chapters_match(
    problem_chapter: str | None,
    chunk_chapter: str | None,
    mode: str = "prefix",
) -> bool:
    problem_raw = str(problem_chapter or "").strip().lower()
    chunk_raw = str(chunk_chapter or "").strip().lower()
    if not problem_raw or not chunk_raw:
        return False
    if mode == "prefix":
        return chunk_raw.startswith(problem_raw)
    if mode == "numeric_stem":
        problem_stem = chapter_numeric_stem(problem_raw)
        chunk_stem = chapter_numeric_stem(chunk_raw)
        if problem_stem and chunk_stem:
            return problem_stem == chunk_stem
        return chunk_raw.startswith(problem_raw)
    raise ValueError(f"unknown chapter match mode: {mode}")


def chunk_quality_issues(text: str) -> list[str]:
    issues: list[str] = []
    if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
        issues.append("control_chars")
    if "\ufffd" in text:
        issues.append("replacement_char")
    if text.count("$") >= 3 and text.count("$") % 2 == 1:
        issues.append("odd_dollar_count")
    if abs(text.count("{") - text.count("}")) >= 8:
        issues.append("brace_imbalance")
    long_tokens = [tok for tok in re.findall(r"\S+", text) if len(tok) >= 45]
    if len(long_tokens) >= 3:
        issues.append("many_long_tokens")
    return issues


def filter_chunks_by_quality(
    chunks: Iterable[Chunk],
    mode: str = "none",
) -> list[Chunk]:
    if mode == "none":
        return list(chunks)
    if mode != "light":
        raise ValueError(f"unknown quality filter mode: {mode}")

    out: list[Chunk] = []
    for chunk in chunks:
        if chunk_quality_issues(chunk.text):
            continue
        out.append(chunk)
    return out


def filter_by_author_chapter(
    chunks: Iterable[Chunk],
    author: Optional[str] = None,
    chapter_prefix: Optional[str] = None,
    chapter_match_mode: str = "prefix",
) -> list[Chunk]:
    out = []
    for c in chunks:
        if author and c.author != author:
            continue
        if chapter_prefix and not chapters_match(
            chapter_prefix,
            str(c.chapter),
            mode=chapter_match_mode,
        ):
            continue
        out.append(c)
    return out
