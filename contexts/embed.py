"""Embedding utilities for RAG retrieval.

Uses OpenAI `text-embedding-3-large` (3072-d). Embeddings are cached on disk
per (model, text-hash) so re-runs are cheap. One cache file per corpus.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

DEFAULT_MODEL = "text-embedding-3-large"
BATCH_SIZE = 64


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        key_path = Path.home() / ".openai_key"
        if key_path.exists():
            key = key_path.read_text().strip()
    if not key:
        raise RuntimeError("No OpenAI API key found (set OPENAI_API_KEY or ~/.openai_key)")
    return key


def _text_hash(text: str, model: str) -> str:
    return hashlib.sha1(f"{model}|{text}".encode()).hexdigest()


def _post(api_key: str, payload: dict, max_retries: int = 6) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers, json=payload, timeout=60,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            status = r.status_code if "r" in dir() else None
            if status == 429 or (status and 500 <= status < 600):
                wait = min(2 ** (attempt + 2), 60)
                print(f"  embed retry {attempt+1}/{max_retries} after {wait}s (HTTP {status})")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("embed: exhausted retries")


def embed_texts(texts: list[str], cache_path: Path, model: str = DEFAULT_MODEL) -> np.ndarray:
    """Return [N, D] float32 embedding array, caching by text hash."""
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            corrupt_path = cache_path.with_suffix(cache_path.suffix + f".corrupt-{int(time.time())}")
            cache_path.rename(corrupt_path)
            print(f"embedding cache was malformed; moved to {corrupt_path}", flush=True)
            cache = {}

    missing_idx: list[int] = []
    missing_texts: list[str] = []
    hashes = [_text_hash(t, model) for t in texts]
    for i, h in enumerate(hashes):
        if h not in cache:
            missing_idx.append(i)
            missing_texts.append(texts[i])

    if missing_texts:
        api_key = _api_key()
        print(
            f"embedding {len(missing_texts)}/{len(texts)} texts into {cache_path}",
            flush=True,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(missing_texts), BATCH_SIZE):
            batch = missing_texts[start : start + BATCH_SIZE]
            resp = _post(api_key, {"model": model, "input": batch})
            for j, emb in enumerate(resp["data"]):
                h = hashes[missing_idx[start + j]]
                cache[h] = emb["embedding"]
            done = min(start + len(batch), len(missing_texts))
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(cache, f)
            tmp_path.replace(cache_path)
            print(f"  embedded {done}/{len(missing_texts)}", flush=True)

    dim = len(next(iter(cache.values())))
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, h in enumerate(hashes):
        out[i] = np.asarray(cache[h], dtype=np.float32)
    return out
