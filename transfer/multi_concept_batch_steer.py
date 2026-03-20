#!/usr/bin/env python3
"""
Run transferred steering for *many* concepts over the same test prompts (one target
model load). Skips concepts missing directions or W.pkl.

Prereq per concept: same as transfer-one (source .pkl, paired .npz, merge -> W_pkl).

Example:
  python transfer/multi_concept_batch_steer.py \\
    --source_model llama_3.0_8b --target_model llama_3.1_8b_hf \\
    -c fears -t max_attn_per_layer -l soft \\
    --concepts fire,bathing,heights,spiders \\
    --coef 0.7 --overwrite \\
    --out_jsonl data/transfer_runs/multi_concept_fears_v1-5.jsonl
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
from transfer.transfer_utils import (
    ensure_repo_cwd,
    layer_indices_steered,
    w_pkl_path,
)


def _models():
    return [
        "llama_3.0_8b",
        "llama_3.1_8b",
        "llama_3.1_8b_hf",
        "llama_3.3_8b",
        "llama_3.1_70b",
        "llama_3.3_70b",
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Multi-concept transferred batch eval -> JSONL.")
    p.add_argument("--source_model", required=True, choices=_models())
    p.add_argument("--target_model", required=True, choices=_models())
    p.add_argument("--concept_type", "-c", required=True)
    p.add_argument(
        "--concepts",
        default=None,
        help="Comma-separated concept strings (must match data/concepts/<type>.txt after lower/trim).",
    )
    p.add_argument(
        "--first_n",
        type=int,
        default=None,
        help="Use first N concepts from sorted concepts file (same order as utils.read_file).",
    )
    p.add_argument("--rep_token", "-t", default="max_attn_per_layer")
    p.add_argument("--label", "-l", default="soft", choices=["soft", "hard"])
    p.add_argument("--control_method", "-cm", default="rfm")
    p.add_argument("--coef", type=float, default=0.7)
    p.add_argument("--max_tokens", type=int, default=150)
    p.add_argument("--start_from_token", type=int, default=0)
    p.add_argument("--yaml_path", default="test_prompts.yaml")
    p.add_argument("--versions", default="1,2,3,4,5")
    p.add_argument("--out_jsonl", required=True)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Truncate out_jsonl at start (default is append).",
    )
    p.add_argument(
        "--skip_baseline",
        action="store_true",
    )
    return p.parse_args()


def _concept_list(args) -> list[str]:
    import utils

    dataset_to_lower = {
        "fears": True,
        "personalities": True,
        "moods": True,
        "places": False,
        "personas": False,
        "jailbreaking": False,
        "custom": False,
    }
    fname = f"data/concepts/{args.concept_type}.txt"
    all_c = utils.read_file(fname, lower=dataset_to_lower[args.concept_type])
    if args.concepts:
        want = [x.strip() for x in args.concepts.split(",") if x.strip()]
        if dataset_to_lower.get(args.concept_type):
            want = [w.lower() for w in want]
        missing = [w for w in want if w not in all_c]
        if missing:
            raise ValueError(f"Unknown concepts (not in {fname}): {missing}")
        return want
    if args.first_n is not None:
        return all_c[: args.first_n]
    raise ValueError("Provide --concepts a,b,c or --first_n N")


def main():
    ensure_repo_cwd()
    args = parse_args()

    try:
        rep_token = int(args.rep_token)
    except ValueError:
        rep_token = args.rep_token

    use_soft_labels = args.label == "soft"
    versions = [int(x.strip()) for x in args.versions.split(",") if x.strip()]
    concepts = _concept_list(args)

    yaml_path = Path(args.yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        test_prompts_dict = yaml.safe_load(f)
    sub = test_prompts_dict[args.concept_type]

    import utils
    from neural_controllers import NeuralController

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.is_file():
        out_path.unlink()

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

    n_ok = 0
    for concept in concepts:
        vec_path = utils.get_concept_vec_filename(
            args.control_method,
            concept,
            rep_token,
            args.source_model,
            use_soft_labels,
        )
        wp = w_pkl_path(
            args.source_model,
            args.target_model,
            args.concept_type,
            concept,
        )
        if not os.path.isfile(vec_path):
            print(f"[skip] no source directions: {concept} -> {vec_path}", flush=True)
            continue
        if not wp.is_file():
            print(f"[skip] no W pickle: {concept} -> {wp}", flush=True)
            continue

        with open(vec_path, "rb") as f:
            src_dirs = pickle.load(f)
        with open(wp, "rb") as f:
            W_layers = pickle.load(f)

        transferred = apply_transfer(W_layers, src_dirs)
        controller.individual_directions = transferred
        controller.get_all_directions([])

        for v in versions:
            if v not in sub:
                raise KeyError(f"Version {v} missing under {args.concept_type}")
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
                "concept": concept,
                "version": v,
                "prompt": prompt,
                "coef": args.coef,
                "rep_token": rep_token,
                "source_model": args.source_model,
                "target_model": args.target_model,
                "w_pkl": str(wp),
                "baseline": baseline_text,
                "steered": steered_text,
            }
            with open(out_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[ok] {concept} -> {len(versions)} rows in {out_path}", flush=True)
        n_ok += 1

    del llm, model, controller
    torch.cuda.empty_cache()
    print(f"Done. Concepts with data: {n_ok}/{len(concepts)}", flush=True)


if __name__ == "__main__":
    main()
