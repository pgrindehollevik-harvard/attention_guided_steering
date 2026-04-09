#!/usr/bin/env python3
"""
Pretty-print transfer batch JSONL for scrolling (pipe to less).

  python transfer/preview_jsonl.py data/transfer_runs/fire_maxattn_v1-5.jsonl | less -R

Export for collaborators (no API — static HTML or LaTeX, same layout as terminal):

  python transfer/preview_jsonl.py results/transfer_runs/exp.jsonl \\
    --out-html results/transfer_runs/exp_readable.html \\
    --out-tex results/transfer_runs/exp_readable.tex

GPT scores + HTML with judge text: use transfer/evaluate_jsonl.py --out_report (needs OPENAI_API_KEY).

Use / in less to search; q to quit.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transfer.transfer_utils import ensure_repo_cwd


def _iter_rows(jsonl_path: Path):
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)


def _emit_blocks(r: dict, i: int, no_baseline: bool, write):
    sep = "=" * 72
    sub = "-" * 72
    write(sep + "\n")
    write(
        f"ROW {i + 1}  |  concept={r.get('concept')}  |  version={r.get('version')}  |  coef={r.get('coef')}\n"
    )
    write(f"rep_token={r.get('rep_token')}\n")
    write(sub + "\n")
    write("PROMPT:\n")
    write((r.get("prompt", "") or "") + "\n\n")
    if not no_baseline and r.get("baseline") is not None:
        write(sub + "\nBASELINE (no steering)\n" + sub + "\n")
        write(str(r["baseline"]) + "\n\n")
    if r.get("native_source_steered"):
        write(sub + "\nNATIVE SOURCE STEERED\n" + sub + "\n")
        write(str(r["native_source_steered"]) + "\n\n")
    if r.get("native_target_steered"):
        write(sub + "\nNATIVE TARGET STEERED\n" + sub + "\n")
        write(str(r["native_target_steered"]) + "\n\n")
    if r.get("transfer_target_steered"):
        write(sub + "\nTRANSFER TARGET STEERED\n" + sub + "\n")
        write(str(r["transfer_target_steered"]) + "\n\n")
    if r.get("steered") and not r.get("transfer_target_steered"):
        write(sub + "\nSTEERED\n" + sub + "\n")
        write(str(r.get("steered", "")) + "\n\n")


def _write_html(path: Path, jsonl_path: Path, no_baseline: bool) -> None:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(jsonl_path.name)}</title>",
        "<style>",
        "body{font-family:ui-monospace,Menlo,Consolas,monospace;max-width:920px;margin:2rem auto;line-height:1.5;font-size:14px;}",
        "pre{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;border:1px solid #d0d7de;padding:1rem;border-radius:6px;}",
        "h2{font-family:system-ui,sans-serif;font-size:1rem;margin:2rem 0 .5rem;color:#24292f;}",
        ".meta{color:#57606a;font-size:12px;margin-bottom:1rem;}",
        "</style></head><body>",
        f"<p class='meta'>Generated from <code>{html.escape(str(jsonl_path))}</code> (static preview — no GPT).</p>",
    ]
    for i, r in _iter_rows(jsonl_path):
        buf: list[str] = []

        def write(s: str) -> None:
            buf.append(s)

        _emit_blocks(r, i, no_baseline, write)
        parts.append(f"<h2>Row {i + 1}</h2>")
        parts.append("<pre>")
        parts.append(html.escape("".join(buf)))
        parts.append("</pre>")
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _latex_escape(s: str) -> str:
    out: list[str] = []
    for c in s:
        if c in "{}\\$%&#^_~":
            out.append(
                {
                    "{": r"\{",
                    "}": r"\}",
                    "\\": r"\textbackslash{}",
                    "$": r"\$",
                    "%": r"\%",
                    "&": r"\&",
                    "#": r"\#",
                    "^": r"\textasciicircum{}",
                    "_": r"\_",
                    "~": r"\textasciitilde{}",
                }[c]
            )
        else:
            out.append(c)
    return "".join(out)


def _write_tex(path: Path, jsonl_path: Path, no_baseline: bool) -> None:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{url}",
        r"\begin{document}",
        r"\title{Steering outputs (static export)}",
        r"\maketitle",
        r"\noindent\textit{Source:} \texttt{"
        + _latex_escape(str(jsonl_path))
        + r"} (no GPT judge in this file).",
        r"\par\medskip",
    ]
    for i, r in _iter_rows(jsonl_path):
        buf: list[str] = []

        def write(s: str) -> None:
            buf.append(s)

        _emit_blocks(r, i, no_baseline, write)
        text = "".join(buf)
        lines.append(r"\section*{Row " + str(i + 1) + r"}")
        lines.append(r"\begin{verbatim}")
        lines.append(text.rstrip("\n"))
        lines.append(r"\end{verbatim}")
        lines.append("")
    lines.append(r"\end{document}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_repo_cwd()
    p = argparse.ArgumentParser(description="Print batch JSONL in a readable form.")
    p.add_argument("jsonl", type=Path, help="Path to .jsonl from batch_steer_transferred.py")
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not print baseline (if you used --skip_baseline).",
    )
    p.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Write static HTML (same layout as terminal; open in browser).",
    )
    p.add_argument(
        "--out-tex",
        type=Path,
        default=None,
        help="Write LaTeX source (compile with pdflatex/xelatex; avoid \\end{verbatim} in model text).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print to stdout (use with --out-html / --out-tex).",
    )
    args = p.parse_args()

    if not args.jsonl.is_file():
        print(f"Not found: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    if args.out_html:
        _write_html(args.out_html, args.jsonl, args.no_baseline)
        print(f"Wrote HTML -> {args.out_html}", file=sys.stderr)
    if args.out_tex:
        _write_tex(args.out_tex, args.jsonl, args.no_baseline)
        print(f"Wrote LaTeX -> {args.out_tex}", file=sys.stderr)

    if not args.quiet:
        for i, r in _iter_rows(args.jsonl):

            def write(s: str) -> None:
                sys.stdout.write(s)

            _emit_blocks(r, i, args.no_baseline, write)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Pipe reader (e.g. `less` after `q`) closed; avoid noisy traceback.
        try:
            sys.stdout.close()
        except BrokenPipeError:
            pass
        raise SystemExit(0)
