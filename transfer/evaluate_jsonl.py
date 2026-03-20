#!/usr/bin/env python3
"""
Evaluate rows from batch_steer_transferred / multi_concept_batch_steer JSONL.

1) **metrics** (default, no API): length + simple repetition heuristic on assistant text.
2) **openai**: GPT judge using the same templates as 3_evaluate_steered_outputs.py
   (data/evaluation_prompts/phobia_eval_v{version}.txt for fears). Requires OPENAI_API_KEY.

Example:
  python transfer/evaluate_jsonl.py \\
    --in_jsonl data/transfer_runs/colleague_demo_fears.jsonl \\
    --out_csv data/transfer_runs/colleague_demo_fears_eval.csv \\
    --mode both
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transfer.transfer_utils import ensure_repo_cwd


def extract_llama3_assistant(decoded: str) -> str:
    """Strip chat template preamble; keep assistant reply text."""
    if not decoded:
        return ""
    for marker in ("<|start_header_id|>assistant", "assistant"):
        if marker in decoded:
            rest = decoded.split(marker, 1)[-1]
            rest = rest.replace("<|end_header_id|>", "", 1).lstrip("\n ")
            break
    else:
        rest = decoded
    eot = "<|eot_id|>"
    if eot in rest:
        rest = rest.split(eot)[0]
    return rest.strip()


def repetition_score(text: str) -> float:
    """Higher = more repeated words (0..1 rough)."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 8:
        return 0.0
    c = Counter(words)
    top, _ = c.most_common(1)[0]
    return top / len(words)


def parse_gpt_score(content: str) -> tuple[int, str]:
    score = 0
    if "Score: " in content:
        try:
            score = int(content.split("Score: ", 1)[1].lstrip()[0])
        except (IndexError, ValueError):
            score = 0
    return score, content


def main():
    ensure_repo_cwd()
    p = argparse.ArgumentParser(description="Evaluate JSONL from transfer batch runs.")
    p.add_argument("--in_jsonl", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument(
        "--mode",
        choices=["metrics", "openai", "both"],
        default="metrics",
        help="metrics=no API; openai=GPT judge (fears templates); both.",
    )
    p.add_argument(
        "--concept_type",
        default="fears",
        help="For openai: which evaluation prompt family (default fears -> phobia_eval_v{version}).",
    )
    p.add_argument(
        "--gpt_model",
        default="gpt-4o-2024-11-20",
        help="OpenAI chat model id (match 3_evaluate_steered_outputs.py if you want parity).",
    )
    args = p.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    client = None
    if args.mode in ("openai", "both"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY required for openai/both mode.")
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    import utils

    rows_out = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            concept = r.get("concept", "")
            version = int(r.get("version", 0))
            coef = r.get("coef", "")
            baseline_raw = r.get("baseline") or ""
            steered_raw = r.get("steered") or ""

            b_txt = extract_llama3_assistant(baseline_raw)
            s_txt = extract_llama3_assistant(steered_raw)

            rec = {
                "concept": concept,
                "version": version,
                "coef": coef,
                "baseline_chars": len(b_txt),
                "steered_chars": len(s_txt),
                "steered_repeat_frac": round(repetition_score(s_txt), 4),
                "baseline_repeat_frac": round(repetition_score(b_txt), 4),
            }

            if args.mode in ("openai", "both") and client is not None:
                if args.concept_type != "fears":
                    raise NotImplementedError(
                        "openai mode currently wires data/evaluation_prompts/phobia_eval_*; "
                        f"extend for concept_type={args.concept_type!r}"
                    )
                template = utils.load_prompt("fears", str(version))
                user_prompt = template.format(personality=concept, parsed_response=s_txt)
                out = client.chat.completions.create(
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.0,
                    max_tokens=120,
                    model=args.gpt_model,
                )
                content = out.choices[0].message.content or ""
                gpt_score, _ = parse_gpt_score(content)
                rec["gpt_steered_score"] = gpt_score
                rec["gpt_steered_raw"] = content.replace("\n", " ")[:500]

            rows_out.append(rec)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_out:
        raise ValueError("No rows read from JSONL.")

    fieldnames = list(rows_out[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
