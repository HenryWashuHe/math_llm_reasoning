"""GPT-4o binary judge using the OpenAI Batch API (50% cheaper, 24h SLA).

Input:  generations JSONL with rows {problem_id, condition, question,
        reference_solution, model_answer}
Output: judgments JSONL with rows {problem_id, condition, label ∈
        {"correct","incorrect"}, note}

Two-step flow:
  submit  — build .jsonl batch input, upload to /v1/files, create batch,
            write batch_id to a marker file.
  collect — poll batch status, download output when complete, parse into
            judgments JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

JUDGE_MODEL = "gpt-4o"

JUDGE_PROMPT = """\
You are grading a student's solution to a linear algebra problem.

You will see the problem, the reference solution, and the student's answer.
Decide whether the student's final answer is mathematically correct. Ignore
presentation / LaTeX formatting issues; judge only correctness of the math.

Reply with STRICT JSON:
  {"label": "correct" | "incorrect", "note": "<one sentence>"}

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


def _build_request(record: dict, custom_id: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        problem=record["question"],
        reference=record["reference_solution"],
        answer=record["model_answer"],
    )
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 200,
        },
    }


def cmd_submit(args: argparse.Namespace) -> None:
    api_key = _api_key()
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    batch_input = args.marker.with_suffix(".batch_input.jsonl")

    n = 0
    with open(args.generations) as fin, open(batch_input, "w") as fout:
        for i, line in enumerate(fin):
            rec = json.loads(line)
            custom_id = f"{rec['problem_id']}__{rec.get('condition','base')}__{i}"
            fout.write(json.dumps(_build_request(rec, custom_id)) + "\n")
            n += 1
    print(f"prepared {n} requests at {batch_input}")

    with open(batch_input, "rb") as f:
        r = requests.post(
            "https://api.openai.com/v1/files",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("batch.jsonl", f, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
    r.raise_for_status()
    file_id = r.json()["id"]

    r = requests.post(
        "https://api.openai.com/v1/batches",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        timeout=60,
    )
    r.raise_for_status()
    batch_id = r.json()["id"]

    with open(args.marker, "w") as f:
        json.dump({"batch_id": batch_id, "file_id": file_id, "n": n}, f)
    print(f"submitted batch {batch_id} ({n} requests) -> {args.marker}")


def cmd_collect(args: argparse.Namespace) -> None:
    api_key = _api_key()
    with open(args.marker) as f:
        info = json.load(f)
    batch_id = info["batch_id"]

    headers = {"Authorization": f"Bearer {api_key}"}
    while True:
        r = requests.get(f"https://api.openai.com/v1/batches/{batch_id}",
                         headers=headers, timeout=60)
        r.raise_for_status()
        status = r.json()["status"]
        print(f"  batch {batch_id}: {status}")
        if status == "completed":
            out_file_id = r.json()["output_file_id"]
            break
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"batch ended in {status}")
        if not args.wait:
            print("(not waiting; re-run with --wait)")
            return
        time.sleep(args.poll_interval)

    r = requests.get(f"https://api.openai.com/v1/files/{out_file_id}/content",
                     headers=headers, timeout=300)
    r.raise_for_status()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_err = 0, 0
    with open(args.out, "w") as fout:
        for line in r.text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["custom_id"]
            pid, condition, _ = cid.split("__", 2)
            body = row.get("response", {}).get("body")
            if not body:
                n_err += 1
                fout.write(json.dumps({
                    "problem_id": pid, "condition": condition,
                    "label": "error", "note": "no response body",
                }) + "\n")
                continue
            try:
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                label = parsed.get("label", "error")
                note = parsed.get("note", "")
            except Exception as e:
                label, note = "error", f"parse: {e}"
            fout.write(json.dumps({
                "problem_id": pid, "condition": condition,
                "label": label, "note": note,
            }) + "\n")
            n_ok += 1 if label in ("correct", "incorrect") else 0
            n_err += 0 if label in ("correct", "incorrect") else 1

    print(f"wrote {args.out}  (labeled={n_ok}, errors={n_err})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("submit")
    s1.add_argument("--generations", type=Path, required=True)
    s1.add_argument("--marker", type=Path, required=True)
    s1.set_defaults(func=cmd_submit)

    s2 = sub.add_parser("collect")
    s2.add_argument("--marker", type=Path, required=True)
    s2.add_argument("--out", type=Path, required=True)
    s2.add_argument("--wait", action="store_true")
    s2.add_argument("--poll-interval", type=int, default=120)
    s2.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
