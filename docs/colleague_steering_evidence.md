# Colleague demo — talking points

Short supplement to **`transfer/README.md`** (commands and timing live there).

**What to share:** HTML report from `evaluate_jsonl.py --out_report`, optional JSONL/CSV, git **branch + commit**.

**Expectations**

- Steering shifts **style / affect** more than forcing a **keyword** (e.g. “fire”) unless the **question** invites it.
- **Coefficient ~0.65–0.8** is usually readable; **≥0.9** often **degenerates** (loops, random context) — fine as a stress test, not “best” quality.
- **Transfer** (mapped directions) can look **softer** than **native** steering on the source model.
- **Max-attn** directions vs **`-1`**: max-attn matches paper-style readout; `-1` matches current activation collection for \(W\).

**If multi-concept JSONL only has one concept:** others were **skipped** (missing `data/directions/…pkl` or `W_*_W.pkl`). Run the per-concept build (or `run_full_pipeline_many.sh`) for each name.

**GPT judge:** same templates as `3_evaluate_steered_outputs.py` (`data/evaluation_prompts/phobia_eval_v*.txt`). Extra prompts from a file use **v1** rubric by default — interpret scores cautiously if the question differs a lot from that template.
