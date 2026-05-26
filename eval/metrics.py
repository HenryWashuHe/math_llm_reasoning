"""Aggregate judgment rows into per-arm metrics with CIs.

Input: a JSONL where each row has at minimum
  problem_id, condition, label  (label in {"correct", "incorrect", "unknown"})
Optional fields: task_family, arm_name, baseline_arm.

The `condition` string is expected to look like
  "<context>__<arm>"  e.g. "retrieved_context__ablate"
so we can slice metrics by context and arm independently. If your condition
strings don't follow that convention, pass --arm-key / --context-key to
specify alternate fields explicitly.

Computes per (context, arm, task_family):
  n, n_correct, acc, acc_ci_lo, acc_ci_hi   (bootstrap over problem_ids)
Per arm vs baseline_arm:
  break_rate = #(baseline_correct & arm_incorrect) / #(baseline_correct)
  fisher_p   = Fisher exact test of break contingency vs random-seed arms

Writes a markdown summary and a JSON blob.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _bootstrap_ci(values: list[int], n_boot: int = 2000, alpha: float = 0.05,
                  seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return (lo, hi)


def _fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for 2x2 table [[a,b],[c,d]]."""
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        return float(p)
    except ImportError:
        pass

    def logfact(n: int) -> float:
        return math.lgamma(n + 1)

    def prob(a_: int, r1: int, r2: int, c1: int, N: int) -> float:
        b_ = r1 - a_; c_ = c1 - a_; d_ = r2 - c_
        if min(a_, b_, c_, d_) < 0:
            return 0.0
        return math.exp(
            logfact(r1) + logfact(r2) + logfact(c1) + logfact(N - c1)
            - logfact(N) - logfact(a_) - logfact(b_) - logfact(c_) - logfact(d_)
        )

    N = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c
    obs = prob(a, r1, r2, c1, N)
    p = 0.0
    for a_ in range(0, min(r1, c1) + 1):
        cell = prob(a_, r1, r2, c1, N)
        if cell <= obs + 1e-12:
            p += cell
    return min(p, 1.0)


def _split_condition(cond: str) -> tuple[str, str]:
    if "__" in cond:
        ctx, arm = cond.split("__", 1)
        return ctx, arm
    return "", cond


def aggregate(rows: list[dict], baseline_arm: str = "none",
              random_arm_prefix: str = "random_") -> dict:
    by_cohort: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        ctx, arm = _split_condition(r["condition"])
        fam = r.get("task_family") or "all"
        by_cohort[(ctx, arm, fam)].append(r)

    acc_table = []
    for (ctx, arm, fam), rs in sorted(by_cohort.items()):
        labeled = [1 if r["label"] == "correct" else 0 for r in rs if r["label"] in ("correct", "incorrect")]
        n = len(labeled); n_c = sum(labeled)
        lo, hi = _bootstrap_ci(labeled) if labeled else (float("nan"), float("nan"))
        acc_table.append({
            "context": ctx, "arm": arm, "task_family": fam,
            "n": n, "n_correct": n_c,
            "accuracy": n_c / n if n else float("nan"),
            "acc_ci_lo": lo, "acc_ci_hi": hi,
        })

    break_table = []
    for (ctx, arm, fam), rs in sorted(by_cohort.items()):
        if arm == baseline_arm:
            continue
        base_rs = by_cohort.get((ctx, baseline_arm, fam), [])
        base_by_pid = {r["problem_id"]: r["label"] for r in base_rs}
        n_base_correct = 0; n_break = 0
        for r in rs:
            pid = r["problem_id"]
            if base_by_pid.get(pid) == "correct":
                n_base_correct += 1
                if r["label"] == "incorrect":
                    n_break += 1
        break_table.append({
            "context": ctx, "arm": arm, "task_family": fam,
            "n_baseline_correct": n_base_correct,
            "n_break": n_break,
            "break_rate": n_break / n_base_correct if n_base_correct else float("nan"),
        })

    fisher_rows = []
    target_arms = sorted({row["arm"] for row in break_table
                          if not row["arm"].startswith(random_arm_prefix)
                          and row["arm"] != "passthrough"})
    rand_arms = sorted({row["arm"] for row in break_table
                        if row["arm"].startswith(random_arm_prefix)})
    for (ctx, fam) in sorted({(r["context"], r["task_family"]) for r in break_table}):
        for tgt in target_arms:
            t = next((r for r in break_table
                      if r["context"] == ctx and r["arm"] == tgt and r["task_family"] == fam), None)
            if not t:
                continue
            rb_break = 0; rb_total = 0
            for rn in rand_arms:
                rr = next((r for r in break_table
                           if r["context"] == ctx and r["arm"] == rn and r["task_family"] == fam), None)
                if rr:
                    rb_break += rr["n_break"]; rb_total += rr["n_baseline_correct"]
            if t["n_baseline_correct"] == 0 or rb_total == 0:
                p = float("nan")
            else:
                p = _fisher_exact_2x2(
                    t["n_break"], t["n_baseline_correct"] - t["n_break"],
                    rb_break, rb_total - rb_break,
                )
            fisher_rows.append({
                "context": ctx, "arm": tgt, "task_family": fam,
                "target_break": t["n_break"], "target_n": t["n_baseline_correct"],
                "random_break_total": rb_break, "random_n_total": rb_total,
                "fisher_p_two_sided": p,
            })

    return {"accuracy": acc_table, "break_rate": break_table, "fisher_vs_random": fisher_rows}


def _render_md(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# Generation run metrics\n")
    lines.append("## Accuracy\n")
    lines.append("| context | arm | task_family | n | correct | acc | 95% CI |")
    lines.append("|---|---|---|---:|---:|---:|---|")
    for r in summary["accuracy"]:
        lines.append(f"| {r['context']} | {r['arm']} | {r['task_family']} | {r['n']} | {r['n_correct']} | "
                     f"{r['accuracy']:.2f} | [{r['acc_ci_lo']:.2f}, {r['acc_ci_hi']:.2f}] |")
    lines.append("\n## Break rate vs baseline\n")
    lines.append("| context | arm | task_family | breaks / baseline_correct | rate |")
    lines.append("|---|---|---|---:|---:|")
    for r in summary["break_rate"]:
        lines.append(f"| {r['context']} | {r['arm']} | {r['task_family']} | "
                     f"{r['n_break']}/{r['n_baseline_correct']} | {r['break_rate']:.2f} |")
    lines.append("\n## Target arm vs random-feature arms (Fisher 2-sided)\n")
    lines.append("| context | arm | task_family | target | random (pooled) | p |")
    lines.append("|---|---|---|---|---|---:|")
    for r in summary["fisher_vs_random"]:
        lines.append(f"| {r['context']} | {r['arm']} | {r['task_family']} | "
                     f"{r['target_break']}/{r['target_n']} | {r['random_break_total']}/{r['random_n_total']} | "
                     f"{r['fisher_p_two_sided']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", type=Path, required=True,
                    help="JSONL with problem_id, condition, label")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--baseline-arm", default="none")
    ap.add_argument("--random-arm-prefix", default="random_")
    args = ap.parse_args()

    rows: list[dict] = []
    with open(args.judgments) as f:
        for line in f:
            rows.append(json.loads(line))

    summary = aggregate(rows, baseline_arm=args.baseline_arm, random_arm_prefix=args.random_arm_prefix)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_md(summary))
    print(json.dumps({
        "n_rows": len(rows),
        "arms": sorted({r["arm"] for r in summary["accuracy"]}),
        "contexts": sorted({r["context"] for r in summary["accuracy"]}),
        "out_json": str(args.out_json),
        "out_md": str(args.out_md),
    }, indent=2))


if __name__ == "__main__":
    main()
