#!/usr/bin/env python3
"""
aggregate_to_csv.py
===================
Turn the per-run JSONL written by gaia_gt_pilot.py into tidy CSVs for R.

Outputs (both PRIVATE: they contain questions, ground truth and model answers)
-------
1. <out>             LONG format, one row per agent run (task x effort x scaffold x
                     replication). R/00_prepare_data.R reads this file and writes the
                     shareable data/gaia_runs_public.csv.
2. <out>_summary.csv One row per design CELL (task x effort x scaffold).

Derived run-level flags (computed here from the raw trajectory)
---------------------------------------------------------------
* timed_out      : the harness's 240 s timeout fired (scored 0).
* has_attachment : the GAIA task references a file the harness never supplies.
                   Needs --tasks, because old JSONL records do not store it.
* leak_flag      : the (manager-visible) trajectory mentions GAIA or an answer
                   key ("expected answer", "reference solution", ...). This is a
                   heuristic lower bound on benchmark leakage: the web agent's own
                   pages are only seen through its summaries to the manager.

Usage (from the repo root)
--------------------------
    python harness/aggregate_to_csv.py --in raw/full_run.jsonl \
        --tasks data/private/gaia_level1.json --out data/private/gaia_runs.csv
"""

import argparse
import csv
import json
import os
import re
import statistics

# Phrases indicating the agent found GAIA material or a published answer key.
# Deliberately excludes the bare word "benchmark", which matches innocuous
# text such as "the best official benchmark time".
LEAK_PATTERN = re.compile(
    r"\bgaia\b"
    r"|huggingface\.co/datasets"
    r"|\b(?:expected|reference|ground[- ]truth|gold|official) (?:answer|solution)\b"
    r"|\bvalidation (?:set|split)\b",
    re.IGNORECASE,
)

LONG_COLUMNS = [
    "task_id",
    "run_index",
    "model",
    "prompt_variant",      # crossed facet: manager scaffold (tool / code)
    "effort",              # crossed facet: reasoning_effort (low / high)
    "temperature",         # held constant at 1 (not a facet)
    "search_backend",      # held constant; audit trail
    "task_type",
    "level",
    "horizon_steps",       # GAIA annotator step count (intended difficulty)
    "horizon_tools",       # GAIA annotator tool count
    "has_attachment",
    "true_answer",
    "model_answer",
    "correct",             # 1 = GAIA-scored correct, 0 = incorrect
    "wall_time_s",         # end-to-end wall clock
    "n_steps",             # MANAGER memory steps (incl. the initial TaskStep)
    "n_tool_calls",        # MANAGER tool calls (incl. the final_answer call)
    "n_step_errors",       # manager steps that raised a tool/exec error
    "empty_answer",
    "no_tool_use",
    "tool_or_exec_error",
    "no_final_answer",     # agent never emitted a final_answer call
    "harness_exception",
    "crashed",             # 1 if no answer was produced (timeout or exception)
    "timed_out",           # 1 if the crash was the 240 s timeout
    "crash_type",          # none | timeout | generation_error | other
    "crash_msg",
    "leak_flag",
    "timestamp_utc",
    "question",            # kept last: long free text
]


def _flag(flags, name):
    return int(name in (flags or []))


def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_attachment_ids(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    return {str(t.get("task_id")) for t in tasks if t.get("file_name")}


def crash_type(msg):
    if not msg:
        return "none"
    if msg.startswith("TimeoutError"):
        return "timeout"
    if msg.startswith("AgentGenerationError"):
        return "generation_error"
    return "other"


def to_long_row(rec, attach_ids):
    flags = rec.get("error_flags") or []
    traj = rec.get("trajectory") or []
    crash = rec.get("crash")
    if "has_attachment" in rec:
        has_att = int(rec["has_attachment"])
    elif attach_ids is not None:
        has_att = int(str(rec.get("task_id")) in attach_ids)
    else:
        has_att = ""
    traj_text = json.dumps(traj, ensure_ascii=False)
    return {
        "task_id": rec.get("task_id"),
        "run_index": rec.get("run_index"),
        "model": rec.get("model"),
        "prompt_variant": rec.get("prompt_variant"),
        "effort": rec.get("effort"),
        "temperature": rec.get("temperature"),
        "search_backend": rec.get("search_backend"),
        "task_type": rec.get("task_type"),
        "level": rec.get("level"),
        "horizon_steps": rec.get("horizon_steps"),
        "horizon_tools": rec.get("horizon_tools"),
        "has_attachment": has_att,
        "true_answer": rec.get("true_answer"),
        "model_answer": rec.get("model_answer"),
        "correct": rec.get("success"),
        "wall_time_s": rec.get("wall_time_s"),
        "n_steps": rec.get("n_steps"),
        "n_tool_calls": rec.get("n_tool_calls"),
        "n_step_errors": sum(1 for s in traj if s.get("error")),
        "empty_answer": _flag(flags, "empty_answer"),
        "no_tool_use": _flag(flags, "no_tool_use"),
        "tool_or_exec_error": _flag(flags, "tool_or_exec_error"),
        "no_final_answer": _flag(flags, "no_final_answer"),
        "harness_exception": _flag(flags, "harness_exception"),
        "crashed": int(crash is not None),
        "timed_out": int(rec.get("timed_out", crash_type(crash) == "timeout")),
        "crash_type": crash_type(crash),
        "crash_msg": crash,
        "leak_flag": int(bool(LEAK_PATTERN.search(traj_text))),
        "timestamp_utc": rec.get("timestamp_utc"),
        "question": rec.get("question"),
    }


def write_long(rows, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 3) if xs else ""


def write_summary(rows, out_path):
    # One row per design CELL (task x effort x scaffold). Collapsing across
    # conditions would mix them and destroy the within-cell consistency signal.
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["task_id"], r["effort"], r["prompt_variant"]), []).append(r)

    cols = [
        "task_id", "effort", "prompt_variant",
        "task_type", "level", "horizon_steps", "horizon_tools", "has_attachment",
        "true_answer", "n_iterations", "n_correct", "success_rate",
        "consistency",           # max(p, 1-p): 1.0 = always the same outcome
        "n_crashed", "n_timed_out", "n_leak_flag",
        "mean_wall_time_s", "mean_n_steps", "mean_n_tool_calls",
        "n_distinct_answers",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (tid, effort, scaffold), rs in by_cell.items():
            n = len(rs)
            p = sum(1 for r in rs if r["correct"] == 1) / n
            w.writerow({
                "task_id": tid, "effort": effort, "prompt_variant": scaffold,
                "task_type": rs[0]["task_type"], "level": rs[0]["level"],
                "horizon_steps": rs[0]["horizon_steps"],
                "horizon_tools": rs[0]["horizon_tools"],
                "has_attachment": rs[0]["has_attachment"],
                "true_answer": rs[0]["true_answer"],
                "n_iterations": n,
                "n_correct": sum(1 for r in rs if r["correct"] == 1),
                "success_rate": round(p, 3),
                "consistency": round(max(p, 1 - p), 3),
                "n_crashed": sum(r["crashed"] for r in rs),
                "n_timed_out": sum(r["timed_out"] for r in rs),
                "n_leak_flag": sum(r["leak_flag"] for r in rs),
                "mean_wall_time_s": _mean([r["wall_time_s"] for r in rs]),
                "mean_n_steps": _mean([r["n_steps"] for r in rs]),
                "mean_n_tool_calls": _mean([r["n_tool_calls"] for r in rs]),
                "n_distinct_answers": len({(r["model_answer"] or "").strip().lower() for r in rs}),
            })


def main():
    ap = argparse.ArgumentParser(description="JSONL -> tidy CSV aggregator")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--tasks", default=None,
                    help="GAIA task JSON (used to derive has_attachment for old records)")
    ap.add_argument("--out", default="data/private/gaia_runs.csv", help="Output long CSV")
    args = ap.parse_args()

    attach_ids = load_attachment_ids(args.tasks)
    rows = [to_long_row(r, attach_ids) for r in load_records(args.inp)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_long(rows, args.out)
    base, ext = os.path.splitext(args.out)
    summary_path = f"{base}_summary{ext or '.csv'}"
    write_summary(rows, summary_path)

    if rows:
        n_ok = sum(1 for r in rows if r["correct"] == 1)
        print(f"[aggregate] {len(rows)} runs across {len({r['task_id'] for r in rows})} tasks")
        print(f"[aggregate] correct {n_ok}/{len(rows)} = {100 * n_ok / len(rows):.1f}%")
        print(f"[aggregate] timed out {sum(r['timed_out'] for r in rows)}, "
              f"leak-flagged {sum(r['leak_flag'] for r in rows)}")
    print(f"[aggregate] wrote {args.out}\n[aggregate] wrote {summary_path}")


if __name__ == "__main__":
    main()
