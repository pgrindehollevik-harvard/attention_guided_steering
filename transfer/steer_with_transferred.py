#!/usr/bin/env python3
"""
Steer the *target* model using directions mapped from the *source* model.

Loads source RFM directions, applies v_tgt = normalize(X.T @ v_src) per layer using
W from merge_and_fit_mapping.py, then runs generation on the target LLM.

Example:
  python transfer/steer_with_transferred.py \\
    --w_pkl data/transfer_mappings/W_llama_3.1_70b_to_llama_3.3_70b_fears_fire_W.pkl \\
    --source_model llama_3.1_70b --target_model llama_3.3_70b \\
    -c fears --concept fire -t max_attn_per_layer -l soft \\
    --prompt "What is the scariest thing in the world?"
"""
from __future__ import annotations

from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import os
import pickle

import numpy as np
import torch

from transfer.transfer_utils import ensure_repo_cwd, layer_indices_steered


def parse_args():
    p = argparse.ArgumentParser(description="Steer target model with transferred directions.")
    p.add_argument("--w_pkl", required=True, help="Pickle of dict layer_idx -> X (d,d) from merge step.")
    p.add_argument("--source_model", required=True, choices=["llama_3.1_70b", "llama_3.3_70b"])
    p.add_argument("--target_model", required=True, choices=["llama_3.1_70b", "llama_3.3_70b"])
    p.add_argument("--concept_type", "-c", required=True)
    p.add_argument("--concept", required=True)
    p.add_argument(
        "--rep_token",
        "-t",
        default="max_attn_per_layer",
        help="Must match source direction filenames (e.g. max_attn_per_layer or -1).",
    )
    p.add_argument("--label", "-l", default="soft", choices=["soft", "hard"])
    p.add_argument("--control_method", "-cm", default="rfm")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max_tokens", type=int, default=100)
    p.add_argument("--coef", type=float, default=None, help="Single coef; default uses get_coefs for target.")
    p.add_argument("--start_from_token", type=int, default=0, help="Steering mask start (match 2_steer.py).")
    return p.parse_args()


def apply_transfer(W_layers: dict, src_dirs: dict) -> dict:
    """Build target-space directions: v_tgt = normalize(X.T @ v_src)."""
    out = {}
    for ell, X in W_layers.items():
        if ell not in src_dirs:
            continue
        v = src_dirs[ell]
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        v = v.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if v.shape[0] != X.shape[0]:
            raise ValueError(f"Layer {ell}: v dim {v.shape[0]} != W row dim {X.shape[0]}")
        mapped = X.T @ v.reshape(-1, 1)
        mapped = mapped.ravel()
        nrm = np.linalg.norm(mapped)
        if nrm < 1e-12:
            raise ValueError(f"Layer {ell}: mapped direction is ~zero")
        mapped = mapped / nrm
        out[ell] = torch.from_numpy(mapped.astype(np.float32))
    missing = set(src_dirs.keys()) - set(out.keys())
    if missing:
        print(f"Warning: no W for layers {sorted(missing)[:10]}... (truncated if many)")
    return out


def main():
    ensure_repo_cwd()
    args = parse_args()

    try:
        rep_token = int(args.rep_token)
    except ValueError:
        rep_token = args.rep_token

    use_soft_labels = args.label == "soft"

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

    coefs = [args.coef] if args.coef is not None else utils.get_coefs(args.target_model, use_soft_labels)

    controller = NeuralController(
        llm,
        llm.tokenizer,
        control_method=args.control_method,
        n_components=1,
        start_from_token=args.start_from_token,
    )
    controller.individual_directions = transferred
    controller.get_all_directions([])

    print("Original (no steering):")
    print(
        controller.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            layers_to_control=[],
            concat_layers_list=[],
            do_sample=False,
        )
    )

    for coef in coefs:
        print(f"\n=== Steered coef={coef} (transferred {args.source_model} -> {args.target_model}) ===")
        out = controller.generate(
            args.prompt,
            hidden_state="block",
            control_coef=coef,
            concat_layers_list=[],
            layers_to_control=layers_to_control,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )
        print(out)


if __name__ == "__main__":
    main()
