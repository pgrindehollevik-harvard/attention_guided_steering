#!/usr/bin/env python3
"""
Pretty-print transfer batch JSONL for scrolling (pipe to less).

  python transfer/preview_jsonl.py data/transfer_runs/fire_maxattn_v1-5.jsonl | less -R

Use / in less to search; q to quit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transfer.transfer_utils import ensure_repo_cwd


def main():
    ensure_repo_cwd()
    p = argparse.ArgumentParser(description="Print batch JSONL in a readable form.")
    p.add_argument("jsonl", type=Path, help="Path to .jsonl from batch_steer_transferred.py")
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not print baseline (if you used --skip_baseline).",
    )
    args = p.parse_args()

    if not args.jsonl.is_file():
        print(f"Not found: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    sep = "=" * 72
    sub = "-" * 72

    with open(args.jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            print(sep)
            print(f"ROW {i + 1}  |  concept={r.get('concept')}  |  version={r.get('version')}  |  coef={r.get('coef')}")
            print(f"rep_token={r.get('rep_token')}")
            print(sub)
            print("PROMPT:")
            print(r.get("prompt", ""))
            print()
            if not args.no_baseline and r.get("baseline") is not None:
                print(sub)
                print("BASELINE (no steering)")
                print(sub)
                print(r["baseline"])
                print()
            if r.get("native_source_steered"):
                print(sub)
                print("NATIVE SOURCE STEERED")
                print(sub)
                print(r["native_source_steered"])
                print()
            if r.get("native_target_steered"):
                print(sub)
                print("NATIVE TARGET STEERED")
                print(sub)
                print(r["native_target_steered"])
                print()
            if r.get("transfer_target_steered"):
                print(sub)
                print("TRANSFER TARGET STEERED")
                print(sub)
                print(r["transfer_target_steered"])
                print()
            if r.get("steered") and not r.get("transfer_target_steered"):
                print(sub)
                print("STEERED")
                print(sub)
                print(r.get("steered", ""))
                print()


if __name__ == "__main__":
    main()
