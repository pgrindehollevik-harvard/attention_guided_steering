#!/usr/bin/env python3
"""
Run transferred steering for one concept over many prompts (e.g. all test_prompts.yaml
entries for fears), write results as JSONL. Loads the model once.

Example:
  python transfer/batch_steer_transferred.py \\
    --w_pkl data/transfer_mappings/W_llama_3.0_8b_to_llama_3.1_8b_hf_fears_fire_W.pkl \\
    --source_model llama_3.0_8b --target_model llama_3.1_8b_hf \\
    -c fears --concept fire -t max_attn_per_layer -l soft \\
    --coef 0.7 --max_tokens 150 \\
    --out_jsonl data/transfer_runs/fire_llama31_maxattn_v1-5.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from transfer.steer_with_transferred import apply_transfer
from transfer.transfer_utils import ensure_repo_cwd, layer_indices_steered


def parse_args():
    p = argparse.ArgumentParser(description="Batch transferred steer; save JSONL.")
    p.add_argument("--w_pkl", required=True)
    _m = [
        "llama_3.0_8b",
        "llama_3.1_8b",
        "llama_3.1_8b_hf",
        "llama_3.3_8b",
        "llama_3.1_70b",
        "llama_3.3_70b",
    ]
    p.add_argument("--source_model", required=True, choices=_m)
    p.add_argument("--target_model", required=True, choices=_m)
    p.add_argument("--concept_type", "-c", required=True, help="YAML top-level key, e.g. fears")
    p.add_argument("--concept", required=True)
    p.add_argument("--rep_token", "-t", default="max_attn_per_layer")
    p.add_argument("--label", "-l", default="soft", choices=["soft", "hard"])
    p.add_argument("--control_method", "-cm", default="rfm")
    p.add_argument("--coef", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=150)
    p.add_argument("--start_from_token", type=int, default=0)
    p.add_argument(
        "--yaml_path",
        default="test_prompts.yaml",
        help="Repo-relative path to test prompts (same format as 2_steer.py).",
    )
    p.add_argument(
        "--versions",
        default="1,2,3,4,5",
        help="Comma-separated test prompt version ids (YAML keys under concept_type).",
    )
    p.add_argument("--out_jsonl", required=True, help="Append one JSON object per prompt.")
    p.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Only run steered pass (faster if you only need steered text).",
    )
    return p.parse_args()


def main():
    ensure_repo_cwd()
    args = parse_args()

    try:
        rep_token = int(args.rep_token)
    except ValueError:
        rep_token = args.rep_token

    use_soft_labels = args.label == "soft"
    versions = [int(x.strip()) for x in args.versions.split(",") if x.strip()]

    yaml_path = Path(args.yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        test_prompts_dict = yaml.safe_load(f)

    if args.concept_type not in test_prompts_dict:
        raise KeyError(f"No key {args.concept_type!r} in {yaml_path}")

    import utils
    from neural_controllers import NeuralController

    vec_path = utils.get_concept_vec_filename(
        args.control_method,
        args.concept,
        rep_token,
        args.source_model,
        use_soft_labels,
    )
    if not os.path.isfile(vec_path):
        raise FileNotFoundError(f"Source directions not found: {vec_path}")

    with open(vec_path, "rb") as f:
        src_dirs = pickle.load(f)
    with open(args.w_pkl, "rb") as f:
        W_layers = pickle.load(f)

    transferred = apply_transfer(W_layers, src_dirs)

    llm = utils.select_llm(args.target_model)
    model = llm.language_model
    layers_to_control = layer_indices_steered(model.config.num_hidden_layers)

    controller = NeuralController(
        llm,
        llm.tokenizer,
        control_method=args.control_method,
        n_components=1,
        start_from_token=args.start_from_token,
    )
    controller.individual_directions = transferred
    controller.get_all_directions([])

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sub = test_prompts_dict[args.concept_type]
    for v in versions:
        if v not in sub:
            raise KeyError(f"Version {v} not in {args.concept_type} in {yaml_path}; keys: {list(sub.keys())}")
        prompt = sub[v]
        if not isinstance(prompt, str):
            prompt = str(prompt)

        baseline_text = None
        if not args.skip_baseline:
            baseline_text = controller.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                layers_to_control=[],
                concat_layers_list=[],
                do_sample=False,
            )

        steered_text = controller.generate(
            prompt,
            hidden_state="block",
            control_coef=args.coef,
            concat_layers_list=[],
            layers_to_control=layers_to_control,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )

        row = {
            "concept_type": args.concept_type,
            "concept": args.concept,
            "version": v,
            "prompt": prompt,
            "coef": args.coef,
            "rep_token": rep_token,
            "source_model": args.source_model,
            "target_model": args.target_model,
            "w_pkl": args.w_pkl,
            "baseline": baseline_text,
            "steered": steered_text,
        }
        with open(out_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[batch] wrote version={v} -> {out_path}", flush=True)

    del llm, model, controller
    torch.cuda.empty_cache()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
