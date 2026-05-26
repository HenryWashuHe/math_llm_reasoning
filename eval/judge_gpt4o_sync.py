"""Synchronous gpt-4o judge — instant turnaround for small cohorts.

Same prompt as judge_gpt4o.py (batch version) but uses /v1/chat/completions
directly. Use this when N is small (<200) and you don't want to wait on the
24h batch SLA. Costs 2x the batch price; for 100 rows that's still pennies.

Threaded via a small worker pool so wall time ~= per-call latency, not N×.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

import requests

MODEL = "gpt-4o"

JUDGE_PROMPT = """\
You are grading a student's solution to a linear algebra problem.

You will see the problem, the reference solution, and the student's answer.
Decide whether the student's final answer is mathematically correct. Ignore
presentation / LaTeX formatting issues; judge only correctness of the math.

Reply with STRICT JSON:
  {{"label": "correct" | "incorrect", "note": "<one sentence>"}}

Problem:
{problem}

Reference solution:
{reference}

Student answer:
{answer}
"""


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        kp = Path.home() / ".openai_key"
        if kp.exists():
            key = kp.read_text().strip()
    if not key:
        raise RuntimeError("no OpenAI key")
    return key


def _judge_one(api_key: str, rec: dict, max_retries: int = 8) -> dict:
    prompt = JUDGE_PROMPT.format(
        problem=rec.get("question") or rec.get("problem", ""),
        reference=rec.get("reference_solution") or rec.get("gold_solution", ""),
        answer=rec.get("model_answer", ""),
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 200,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json=body, timeout=120,
            )
            if r.status_code == 429:
                retry_after = int(r.headers.get("Retry-After", "0") or 0)
                wait = max(retry_after, min(2 ** (attempt + 3), 60))
                time.sleep(wait)
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            label = parsed.get("label", "error")
            note = parsed.get("note", "")
            return {
                "problem_id": rec["problem_id"],
                "condition": rec.get("condition", "base"),
                "task_family": rec.get("task_family", ""),
                "label": label,
                "note": note,
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "problem_id": rec["problem_id"],
                    "condition": rec.get("condition", "base"),
                    "task_family": rec.get("task_family", ""),
                    "label": "error",
                    "note": f"{type(e).__name__}: {e}",
                }
            time.sleep(2 ** attempt)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=Path, nargs="+", required=True,
                    help="one or more generations JSONL files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    api_key = _api_key()

    rows: list[dict] = []
    for path in args.generations:
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with open(args.out, "w") as fout, cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_judge_one, api_key, r) for r in rows]
        for fut in cf.as_completed(futs):
            res = fut.result()
            fout.write(json.dumps(res) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 10 == 0:
                print(f"  judged {n_done}/{len(rows)}")
    print(f"wrote {args.out} ({n_done}/{len(rows)} judged)")


if __name__ == "__main__":
    main()
