#!/usr/bin/env python3
"""
Evaluate rows from batch_steer_transferred / multi_concept_batch_steer JSONL.

1) **metrics** (default, no API): length + simple repetition heuristic on assistant text.
2) **openai** / **both**: GPT judge using the **same templates and .format() contract** as
   `3_evaluate_steered_outputs.py` (`utils.load_prompt(concept_type, version)` →
   `phobia_eval_*` / `personality_eval_*` / `mood_eval_*` / `topophile_eval_*` / `persona_eval_*` /
   `custom_eval_*`). The **transfer-steered** assistant text is passed as `parsed_response`
   (parsed like the main pipeline when possible). Requires OPENAI_API_KEY.
   Unsupported: e.g. `jailbreaking` (no `data/evaluation_prompts/*` family yet) — use `--mode metrics`.
3) **--out_report report.html**: human-readable page with prompt, baseline, steered, full GPT
   feedback (implies GPT calls even if --mode metrics).

Example:
  export OPENAI_API_KEY=sk-...
  python transfer/evaluate_jsonl.py \\
    --in_jsonl data/transfer_runs/colleague_demo_fears.jsonl \\
    --out_csv data/transfer_runs/colleague_demo_fears_eval.csv \\
    --out_report data/transfer_runs/colleague_demo_fears_report.html \\
    --mode both
"""
from __future__ import annotations

import argparse
import csv
import html
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
    _word, top_count = c.most_common(1)[0]
    return top_count / len(words)


# Mirrors 3_evaluate_steered_outputs.py + utils.load_prompt supported labels.
GPT_EVAL_CONCEPT_TYPES = frozenset(
    {"fears", "personalities", "moods", "places", "personas", "custom"}
)


def parsed_response_for_gpt_judge(
    transfer_raw: str,
    assistant_stripped: str,
    coef,
    target_model: str,
    utils_mod,
) -> str:
    """
    Same extraction idea as 3_evaluate: utils.parse_personality_responses on full decode.
    Falls back to chat-template stripping if parsing fails or model id is unknown.
    """
    if not (transfer_raw or "").strip():
        return assistant_stripped or ""
    tm = (target_model or "").strip() or "llama_3.1_8b_hf"
    try:
        parsed = utils_mod.parse_personality_responses((coef, transfer_raw), tm)
        parsed = (parsed or "").strip()
        return parsed if parsed else (assistant_stripped or "")
    except Exception:
        return assistant_stripped or ""


def parse_gpt_score(content: str) -> tuple[int, str]:
    score = 0
    if "Score: " in content:
        try:
            score = int(content.split("Score: ", 1)[1].lstrip()[0])
        except (IndexError, ValueError):
            score = 0
    return score, content


def write_html_report(
    path: Path,
    rows: list[dict],
    title: str,
    in_jsonl: str,
    gpt_model: str,
) -> None:
    """Supports legacy (baseline + steered) or triple (baseline + native + transfer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;line-height:1.45;}",
        "h1{font-size:1.25rem;}",
        "section{border:1px solid #ccc;border-radius:8px;padding:1rem;margin-bottom:1.5rem;background:#fafafa;}",
        "h2{margin:0 0 .75rem;font-size:1.05rem;color:#222;}",
        ".step{font-weight:700;color:#0d47a1;margin:1rem 0 .35rem;}",
        ".meta{color:#555;font-size:.9rem;margin-bottom:.5rem;}",
        "pre{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #ddd;padding:.75rem;border-radius:6px;font-size:.88rem;}",
        ".gpt{background:#f0f7ff;border-color:#b3d4fc;}",
        ".score{display:inline-block;padding:.2rem .5rem;border-radius:4px;font-weight:600;}",
        ".s1{background:#d4edda;color:#155724;}",
        ".s0{background:#f8d7da;color:#721c24;}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>Source: <code>{html.escape(in_jsonl)}</code> &middot; GPT model: <code>{html.escape(gpt_model)}</code></p>",
    ]
    for i, r in enumerate(rows, 1):
        sc = int(r.get("gpt_score", -1))
        badge = f"<span class='score s{sc}'>GPT score (transfer): {sc}</span>" if sc in (0, 1) else ""
        triple = r.get("triple")
        pid = r.get("prompt_id", "")
        pid_s = f" &middot; prompt <code>{html.escape(str(pid))}</code>" if pid else ""
        parts.append("<section>")
        parts.append(
            f"<h2>#{i} &mdash; <strong>{html.escape(str(r['concept']))}</strong>"
            f"{pid_s} &middot; eval v{r.get('eval_version', r.get('version', ''))} &middot; coef {html.escape(str(r['coef']))} {badge}</h2>"
        )
        parts.append("<p><strong>User prompt</strong></p>")
        parts.append(f"<pre>{html.escape(r['prompt'])}</pre>")
        parts.append(
            f"<p class='meta'>{html.escape(r.get('metrics_line', ''))}</p>"
        )
        parts.append("<p class='step'>1) Baseline — target model, no steering</p>")
        parts.append(f"<pre>{html.escape(r['b_txt'])}</pre>")
        if triple:
            parts.append(
                f"<p class='step'>2) Native steered — source <code>{html.escape(str(r.get('source_model','')))}</code></p>"
            )
            parts.append(
                f"<pre>{html.escape(r.get('n_txt') or '(missing / skipped)')}</pre>"
            )
            parts.append(
                f"<p class='step'>3) Transfer steered — target <code>{html.escape(str(r.get('target_model','')))}</code></p>"
            )
            parts.append(f"<pre>{html.escape(r.get('t_txt') or '(missing / skipped)')}</pre>")
        else:
            parts.append("<p class='step'>2) Steered (transferred target)</p>")
            parts.append(f"<pre>{html.escape(r.get('s_txt', ''))}</pre>")
        parts.append("<p><strong>GPT judge (transfer steered; full response)</strong></p>")
        parts.append(f"<pre class='gpt'>{html.escape(r.get('gpt_full') or '(no GPT call)')}</pre>")
        if r.get("steered_raw_collapsed"):
            parts.append("<details><summary>Full transfer decode (with template)</summary>")
            parts.append(f"<pre>{html.escape(r['steered_raw_collapsed'])}</pre></details>")
        parts.append("</section>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    ensure_repo_cwd()
    p = argparse.ArgumentParser(description="Evaluate JSONL from transfer batch runs.")
    p.add_argument("--in_jsonl", required=True)
    p.add_argument("--out_csv", default=None, help="Optional metrics/scores CSV.")
    p.add_argument(
        "--out_report",
        default=None,
        help="Optional HTML report (prompt + baseline + steered + full GPT). Implies GPT API calls.",
    )
    p.add_argument(
        "--mode",
        choices=["metrics", "openai", "both"],
        default="metrics",
        help="metrics=no API; openai/both=GPT columns in CSV. --out_report always calls GPT.",
    )
    p.add_argument(
        "--concept_type",
        default="fears",
        help=(
            "For GPT judge: same as 3_evaluate / utils.load_prompt "
            "(fears, personalities, moods, places, personas, custom). "
            "Must match the JSONL run. Unsupported types → use --mode metrics."
        ),
    )
    p.add_argument(
        "--gpt_model",
        default="gpt-4o-2024-11-20",
        help="OpenAI chat model id.",
    )
    args = p.parse_args()

    if not args.out_csv and not args.out_report:
        p.error("Provide at least one of --out_csv or --out_report")

    in_path = Path(args.in_jsonl)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    need_gpt = args.mode in ("openai", "both") or args.out_report is not None
    client = None
    if need_gpt:
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY required for openai/both mode or --out_report.")
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    utils_mod = None
    if need_gpt:
        import utils as utils_mod  # noqa: PLC0415 — same stack as 3_evaluate (heavy)

    rows_out: list[dict] = []
    report_rows: list[dict] = []

    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            concept = r.get("concept", "")
            version = r.get("version", 0)
            try:
                version = int(version)
            except (TypeError, ValueError):
                version = 0
            eval_v = int(r.get("eval_version", version) or 1)
            coef = r.get("coef", "")
            prompt = r.get("prompt") or ""
            prompt_id = r.get("prompt_id", "")
            baseline_raw = r.get("baseline") or r.get("baseline_target") or ""
            transfer_raw = r.get("transfer_target_steered") or r.get("steered") or ""
            native_raw = r.get("native_source_steered")
            triple = "transfer_target_steered" in r or "native_source_steered" in r

            b_txt = extract_llama3_assistant(baseline_raw)
            t_txt = extract_llama3_assistant(transfer_raw)
            n_txt = extract_llama3_assistant(native_raw) if native_raw else ""

            rec = {
                "concept": concept,
                "prompt_id": prompt_id,
                "version": version,
                "eval_version": eval_v,
                "coef": coef,
                "baseline_chars": len(b_txt),
                "steered_chars": len(t_txt),
                "transfer_chars": len(t_txt),
                "steered_repeat_frac": round(repetition_score(t_txt), 4),
                "transfer_repeat_frac": round(repetition_score(t_txt), 4),
                "baseline_repeat_frac": round(repetition_score(b_txt), 4),
            }
            if triple:
                rec["native_chars"] = len(n_txt)
                rec["native_repeat_frac"] = round(repetition_score(n_txt), 4)
                rec["source_model"] = r.get("source_model", "")
                rec["target_model"] = r.get("target_model", "")

            gpt_full = ""
            gpt_score = -1
            if need_gpt and client is not None:
                assert utils_mod is not None
                if args.concept_type not in GPT_EVAL_CONCEPT_TYPES:
                    raise ValueError(
                        "GPT judge uses utils.load_prompt(concept_type, version); "
                        f"no evaluation_prompts family for concept_type={args.concept_type!r}. "
                        f"Supported: {sorted(GPT_EVAL_CONCEPT_TYPES)}. "
                        "Use --mode metrics, or add templates + utils.load_prompt entry."
                    )
                target_model = r.get("target_model", "") or ""
                parsed_for_eval = parsed_response_for_gpt_judge(
                    transfer_raw, t_txt, coef, target_model, utils_mod
                )
                template = utils_mod.load_prompt(args.concept_type, str(eval_v))
                user_prompt = template.format(
                    personality=concept, parsed_response=parsed_for_eval
                )
                out = client.chat.completions.create(
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.0,
                    max_tokens=200,
                    model=args.gpt_model,
                )
                gpt_full = (out.choices[0].message.content or "").strip()
                gpt_score, _ = parse_gpt_score(gpt_full)
                rec["gpt_steered_score"] = gpt_score
                rec["gpt_steered_raw"] = gpt_full.replace("\n", " ")[:500]

            rows_out.append(rec)

            if args.out_report:
                if triple:
                    metrics_line = (
                        f"Chars: baseline {len(b_txt)} | native {len(n_txt)} | transfer {len(t_txt)} — "
                        f"repeat: {rec['baseline_repeat_frac']:.4f} / "
                        f"{rec.get('native_repeat_frac', 0):.4f} / {rec['transfer_repeat_frac']:.4f}"
                    )
                else:
                    metrics_line = (
                        f"Chars: baseline {len(b_txt)} | steered {len(t_txt)} — "
                        f"repeat: {rec['baseline_repeat_frac']:.4f} / {rec['steered_repeat_frac']:.4f}"
                    )
                rr = {
                    "concept": concept,
                    "prompt_id": prompt_id,
                    "version": version,
                    "eval_version": eval_v,
                    "coef": coef,
                    "prompt": prompt if isinstance(prompt, str) else str(prompt),
                    "b_txt": b_txt,
                    "s_txt": t_txt,
                    "t_txt": t_txt,
                    "n_txt": n_txt,
                    "triple": triple,
                    "source_model": r.get("source_model", ""),
                    "target_model": r.get("target_model", ""),
                    "metrics_line": metrics_line,
                    "gpt_score": gpt_score,
                    "gpt_full": gpt_full,
                    "steered_raw_collapsed": transfer_raw[:8000]
                    if len(transfer_raw) > 8000
                    else transfer_raw,
                }
                report_rows.append(rr)

    if not rows_out:
        raise ValueError("No rows read from JSONL.")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        seen: set[str] = set()
        preferred = [
            "concept",
            "prompt_id",
            "version",
            "eval_version",
            "coef",
            "baseline_chars",
            "native_chars",
            "transfer_chars",
            "steered_chars",
            "baseline_repeat_frac",
            "native_repeat_frac",
            "transfer_repeat_frac",
            "steered_repeat_frac",
            "source_model",
            "target_model",
            "gpt_steered_score",
            "gpt_steered_raw",
        ]
        for k in preferred:
            if k in seen:
                continue
            if any(k in row for row in rows_out):
                fieldnames.append(k)
                seen.add(k)
        for row in rows_out:
            for k in row:
                if k not in seen:
                    fieldnames.append(k)
                    seen.add(k)
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows_out)
        print(f"Wrote {len(rows_out)} rows -> {out_path}", flush=True)

    if args.out_report:
        rep_path = Path(args.out_report)
        write_html_report(
            rep_path,
            report_rows,
            title=f"Steering eval: {in_path.name}",
            in_jsonl=str(in_path),
            gpt_model=args.gpt_model,
        )
        print(f"Wrote HTML report -> {rep_path}", flush=True)


if __name__ == "__main__":
    main()
