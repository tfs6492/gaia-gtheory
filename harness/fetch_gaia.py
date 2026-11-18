#!/usr/bin/env python3
"""
fetch_gaia.py
=============
Download the GAIA 2023 validation split from Hugging Face and write the Level 1
tasks to data/private/gaia_level1.json (the format gaia_gt_pilot.py expects).

GAIA is a GATED dataset. Before running this script:
  1. Open https://huggingface.co/datasets/gaia-benchmark/GAIA while logged in
     and accept the access conditions.
  2. Authenticate:  `hf auth login`  (or set the HF_TOKEN environment variable).

The output goes to data/private/, which is git-ignored. Please do not commit or
re-share GAIA questions/answers in plain text: the maintainers ask for this to
keep the benchmark out of training corpora.

Usage (from the repo root)
--------------------------
    python harness/fetch_gaia.py                 # -> data/private/gaia_level1.json
    python harness/fetch_gaia.py --level 2 --out data/private/gaia_level2.json
    python harness/fetch_gaia.py --with-attachments   # also copy attached files
"""

import argparse
import glob
import json
import os
import shutil
import sys

REPO_ID = "gaia-benchmark/GAIA"
SPLIT_DIR = "2023/validation"


def _read_metadata(split_path):
    """Read the split's metadata from either the JSONL or the Parquet layout."""
    jsonl = glob.glob(os.path.join(split_path, "metadata*.jsonl"))
    if jsonl:
        with open(jsonl[0], "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    parquet = glob.glob(os.path.join(split_path, "metadata*.parquet"))
    if parquet:
        import pandas as pd
        return pd.read_parquet(parquet[0]).to_dict(orient="records")
    sys.exit(f"No metadata*.jsonl / metadata*.parquet found under {split_path}")


def _jsonable(x):
    """Parquet rows may carry numpy types; coerce nested values to plain JSON."""
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main():
    ap = argparse.ArgumentParser(description="Fetch the gated GAIA validation set")
    ap.add_argument("--level", default="1", help="GAIA level to keep (1, 2 or 3)")
    ap.add_argument("--out", default="data/private/gaia_level1.json")
    ap.add_argument("--with-attachments", action="store_true",
                    help="Also copy attachment files next to --out (in attachments/). "
                         "Note: the current harness does not pass files to the agent.")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub  (see harness/requirements.txt)")

    try:
        local = snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                                  allow_patterns=[f"{SPLIT_DIR}/*"])
    except Exception as exc:  # gated repo -> 401/403 without accepted terms/token
        sys.exit(f"Download failed ({type(exc).__name__}: {exc}).\n"
                 "Accept the dataset terms on Hugging Face and run `hf auth login`.")

    split_path = os.path.join(local, SPLIT_DIR)
    rows = [r for r in _read_metadata(split_path) if str(r.get("Level")) == str(args.level)]
    rows = _jsonable(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[fetch_gaia] wrote {len(rows)} Level {args.level} tasks -> {args.out}")

    if args.with_attachments:
        att_dir = os.path.join(os.path.dirname(args.out), "attachments")
        os.makedirs(att_dir, exist_ok=True)
        n = 0
        for r in rows:
            if r.get("file_name"):
                src = os.path.join(split_path, r["file_name"])
                if os.path.exists(src):
                    shutil.copy2(src, att_dir)
                    n += 1
        print(f"[fetch_gaia] copied {n} attachments -> {att_dir}")


if __name__ == "__main__":
    main()
