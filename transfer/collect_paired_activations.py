#!/usr/bin/env python3
"""
Collect per-layer hidden states for training prompts (one model load per run).

Run twice (e.g. llama_3.1_70b then llama_3.3_70b, or llama_3.1_8b then llama_3.3_8b) with the same
--concept / --concept_type / --max_prompts, then use merge_and_fit_mapping.py.

Example (70B):
  python transfer/collect_paired_activations.py -m llama_3.1_70b -c fears --concept fire --max_prompts 200 -t -1
  python transfer/collect_paired_activations.py -m llama_3.3_70b -c fears --concept fire --max_prompts 200 -t -1

Example (8B Unsloth 3.1 + community/default 3.3 hub id):
  python transfer/collect_paired_activations.py -m llama_3.1_8b -c fears --concept fire --max_prompts 200 -t -1
  python transfer/collect_paired_activations.py -m llama_3.3_8b -c fears --concept fire --max_prompts 200 -t -1

Example (8B official gated Meta only — Llama 3.0 vs 3.1 Instruct):
  python transfer/collect_paired_activations.py -m llama_3.0_8b -c fears --concept fire --max_prompts 200 -t -1
  python transfer/collect_paired_activations.py -m llama_3.1_8b_hf -c fears --concept fire --max_prompts 200 -t -1
"""
from __future__ import annotations

from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import gc

import numpy as np
import torch
from tqdm import tqdm

from transfer.transfer_utils import (
    default_act_path,
    ensure_repo_cwd,
    layer_indices_steered,
    save_run_meta,
)


def parse_args():
    p = argparse.ArgumentParser(description="Collect activations for transfer (one model load).")
    p.add_argument(
        "--model_name",
        "-m",
        required=True,
        choices=[
            "llama_3.0_8b",
            "llama_3.1_8b",
            "llama_3.1_8b_hf",
            "llama_3.3_8b",
            "llama_3.1_70b",
            "llama_3.3_70b",
        ],
        help="Which model to run (load only this one).",
    )
    p.add_argument(
        "--concept_type",
        "-c",
        required=True,
        choices=["fears", "moods", "personas", "places", "personalities", "custom"],
    )
    p.add_argument(
        "--concept",
        required=True,
        help="Concept string (e.g. fire for fears, or full custom prefix for custom).",
    )
    p.add_argument(
        "--rep_token",
        "-t",
        type=int,
        default=-1,
        help="Token index into sequence (negative = from end). Default -1 = last token.",
    )
    p.add_argument("--max_prompts", type=int, default=200, help="Cap number of training prompts.")
    p.add_argument(
        "--datasize",
        default="single",
        choices=["single", "double", "triple"],
        help="Must match direction extraction (custom_dataset often uses triple).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output .npz path (default under data/paired_activations/).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    ensure_repo_cwd()
    args = parse_args()

    from datasets import training_user_contents_and_labels
    from utils import select_llm

    user_contents, labels = training_user_contents_and_labels(
        args.concept_type,
        args.concept,
        datasize=args.datasize,
        seed=args.seed,
    )
    if args.max_prompts is not None:
        user_contents = user_contents[: args.max_prompts]
        labels = labels[: args.max_prompts]

    print(f"Collecting {len(user_contents)} prompts for model={args.model_name}")

    llm = select_llm(args.model_name)
    model = llm.language_model
    tokenizer = llm.tokenizer
    num_hidden = model.config.num_hidden_layers
    layer_idx_list = layer_indices_steered(num_hidden)
    n_layers = len(layer_idx_list)
    hidden_d = model.config.hidden_size

    n = len(user_contents)
    activations = np.zeros((n, n_layers, hidden_d), dtype=np.float16)
    labels_arr = np.array(labels, dtype=np.float32)

    with torch.inference_mode():
        for i, user_content in enumerate(tqdm(user_contents, desc="forward")):
            chat = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hs = list(out.hidden_states)[1:]  # drop embeddings; hs[ell] = after layer ell

            for j, layer_idx in enumerate(layer_idx_list):
                h = hs[layer_idx][0]  # (seq, d)
                seq_len = h.shape[0]
                tok = args.rep_token if args.rep_token >= 0 else seq_len + args.rep_token
                tok = int(tok)
                vec = h[tok].float().cpu().numpy()
                activations[i, j] = vec.astype(np.float16)

    out_path = args.out
    if out_path is None:
        out_path = str(
            default_act_path(args.model_name, args.concept_type, args.concept)
        )
    meta_path = out_path.replace(".npz", ".meta.json")

    np.savez_compressed(
        out_path,
        activations=activations,
        labels=labels_arr,
        layer_indices=np.array(layer_idx_list, dtype=np.int32),
    )
    save_run_meta(
        Path(meta_path),
        {
            "model_name": args.model_name,
            "concept_type": args.concept_type,
            "concept": args.concept,
            "n_prompts": n,
            "n_layers": n_layers,
            "hidden_size": hidden_d,
            "rep_token": args.rep_token,
            "datasize": args.datasize,
            "npz": out_path,
        },
    )
    print(f"Saved {out_path}")

    del llm, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
