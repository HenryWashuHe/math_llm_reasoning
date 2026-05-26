"""Two-tier judge router: sympy verifier first, gpt-4o batch for the rest.

Reads a generations JSONL, splits rows by whether `task_family` has a
registered sympy verifier and `verifier_meta` is present, writes:
  verified.jsonl    judgments produced locally (label + reason)
  to_judge.jsonl    remaining rows that need GPT-4o judging

Then call eval/judge_gpt4o.py submit/collect on to_judge.jsonl.

This file is just the splitter; it does NOT call any API.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mathreason_sae.eval.sympy_verifier import dispatch


def split_for_judge(generations_path: Path, verified_path: Path, to_judge_path: Path) -> dict:
    n_verified, n_unknown, n_ungated = 0, 0, 0
    label_counts = {"correct": 0, "incorrect": 0, "unknown": 0}

    verified_path.parent.mkdir(parents=True, exist_ok=True)
    to_judge_path.parent.mkdir(parents=True, exist_ok=True)

    with open(generations_path) as fin, \
         open(verified_path, "w") as fv, \
         open(to_judge_path, "w") as fj:
        for line in fin:
            rec = json.loads(line)
            fam = rec.get("task_family")
            meta = rec.get("verifier_meta")
            if not fam or not meta:
                n_ungated += 1
                fj.write(line)
                continue

            try:
                result = dispatch(fam, rec.get("model_answer", ""), meta)
            except Exception as e:
                result = None
                err = str(e)
            if result is None:
                fj.write(line)
                n_ungated += 1
                continue
            if result.label == "unknown":
                fj.write(line)
                n_unknown += 1
                continue

            out = {
                "problem_id": rec["problem_id"],
                "condition": rec.get("condition", "base"),
                "label": result.label,
                "note": result.reason,
                "source": "sympy",
            }
            fv.write(json.dumps(out) + "\n")
            n_verified += 1
            label_counts[result.label] += 1

    return {
        "n_verified": n_verified,
        "n_unknown_fallback": n_unknown,
        "n_ungated_fallback": n_ungated,
        "label_counts": label_counts,
        "verified_path": str(verified_path),
        "to_judge_path": str(to_judge_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=Path, required=True)
    ap.add_argument("--verified", type=Path, required=True)
    ap.add_argument("--to-judge", type=Path, required=True)
    args = ap.parse_args()

    summary = split_for_judge(args.generations, args.verified, args.to_judge)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
