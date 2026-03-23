#!/usr/bin/env python3
"""
Four-way generation (JSONL rows):

  1) **Baseline** — target model, no steering
  2) **Native source** — *source* model + RFM directions trained on the source
  3) **Native target** — *target* model + RFM directions trained on the target (Parmida comparison)
  4) **Transfer** — *target* model + source directions mapped through W

Supports **multiple coefficients**, **many YAML prompt versions**, and an optional
**extra prompts file** (one user message per line, # comments allowed).

Uses **three model load phases** (target → baseline+transfer, unload, source → native,
unload, target → transfer) to reduce peak VRAM vs holding two 8B models at once.
You can use `--keep_target_loaded` to skip the third reload (faster if memory allows).

Example:
  python transfer/batch_triple_compare.py \\
    --source_model llama_3.0_8b --target_model llama_3.1_8b_hf \\
    -c fears -t max_attn_per_layer -l soft \\
    --concepts fire,bathing \\
    --coefs 0.55,0.65,0.75,0.85 \\
    --versions 1,2,3,4,5 \\
    --prompts_file data/transfer_eval_prompts_extra.txt \\
    --overwrite \\
    --out_jsonl data/transfer_runs/triple_fire_bathing.jsonl
"""
from __future__ import annotations

import argparse
import gc
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
from transfer.transfer_utils import ensure_repo_cwd, layer_indices_steered, w_pkl_path


def _models():
    return [
        "llama_3.0_8b",
        "llama_3.1_8b",
        "llama_3.1_8b_hf",
        "llama_3.3_8b",
        "llama_3.1_70b",
        "llama_3.3_70b",
    ]


def load_prompt_list(concept_type: str, yaml_path: Path, versions: list[int], prompts_file: Path | None):
    """List of {id, text, yaml_version_for_eval, eval_version}."""
    out = []
    with open(yaml_path, encoding="utf-8") as f:
        d = yaml.safe_load(f)[concept_type]
    for v in versions:
        if v not in d:
            raise KeyError(f"Version {v} not in {concept_type} in {yaml_path}")
        out.append(
            {
                "id": f"v{v}",
                "text": str(d[v]).strip(),
                "eval_version": v,
            }
        )
    if prompts_file and prompts_file.is_file():
        with open(prompts_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                out.append(
                    {
                        "id": f"extra_{i}",
                        "text": line,
                        "eval_version": 1,
                    }
                )
    return out


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
    raise ValueError("Provide --concepts or --first_n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline + native (source & target) + transfer-steered target JSONL."
    )
    p.add_argument("--source_model", required=True, choices=_models())
    p.add_argument("--target_model", required=True, choices=_models())
    p.add_argument("--concept_type", "-c", required=True)
    p.add_argument("--concepts", default=None)
    p.add_argument("--first_n", type=int, default=None)
    p.add_argument("--rep_token", "-t", default="max_attn_per_layer")
    p.add_argument("--label", "-l", default="soft", choices=["soft", "hard"])
    p.add_argument("--control_method", "-cm", default="rfm")
    p.add_argument(
        "--coefs",
        default="0.55,0.65,0.75,0.85",
        help="Comma-separated steering coefficients.",
    )
    p.add_argument("--max_tokens", type=int, default=150)
    p.add_argument("--start_from_token", type=int, default=0)
    p.add_argument("--yaml_path", default="test_prompts.yaml", type=Path)
    p.add_argument("--versions", default="1,2,3,4,5", help="YAML prompt versions (comma-separated).")
    p.add_argument(
        "--prompts_file",
        default=None,
        type=Path,
        help="Optional extra prompts: one user message per line (# comments ok).",
    )
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--keep_target_loaded",
        action="store_true",
        help="Keep target model in VRAM after baselines (skip reload before transfer). Uses more memory.",
    )
    return p.parse_args()


def _unload(llm):
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ensure_repo_cwd()
    args = parse_args()

    try:
        rep_token = int(args.rep_token)
    except ValueError:
        rep_token = args.rep_token

    use_soft_labels = args.label == "soft"
    versions = [int(x.strip()) for x in args.versions.split(",") if x.strip()]
    coefs = [float(x.strip()) for x in args.coefs.split(",") if x.strip()]
    concepts = _concept_list(args)

    if not args.yaml_path.is_file():
        raise FileNotFoundError(args.yaml_path)

    prompts = load_prompt_list(args.concept_type, args.yaml_path, versions, args.prompts_file)
    if not prompts:
        raise ValueError("No prompts (check --versions and --prompts_file).")

    import utils
    from neural_controllers import NeuralController

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out_path.is_file():
        out_path.unlink()

    # --- Phase 1: target model — baselines (once per unique prompt text) ---
    print("[phase 1] Target model: baselines...", flush=True)
    tgt_llm = utils.select_llm(args.target_model)
    tgt_layers = layer_indices_steered(tgt_llm.language_model.config.num_hidden_layers)
    baseline_ctrl = NeuralController(
        tgt_llm,
        tgt_llm.tokenizer,
        control_method=args.control_method,
        n_components=1,
        start_from_token=args.start_from_token,
    )
    baseline_by_text: dict[str, str] = {}
    for pr in prompts:
        t = pr["text"]
        if t not in baseline_by_text:
            baseline_by_text[t] = baseline_ctrl.generate(
                t,
                max_new_tokens=args.max_tokens,
                layers_to_control=[],
                concat_layers_list=[],
                do_sample=False,
            )

    transfer_cache: dict[tuple[str, str, float], str] = {}
    target_native_cache: dict[tuple[str, str, float], str | None] = {}

    if args.keep_target_loaded:
        print(
            "[phase 1b] Target model: target-native then transfer steered...",
            flush=True,
        )
        tgt_steering_ctrl = NeuralController(
            tgt_llm,
            tgt_llm.tokenizer,
            control_method=args.control_method,
            n_components=1,
            start_from_token=args.start_from_token,
        )
        for concept in concepts:
            # --- Target-native (directions from target checkpoint) ---
            vec_path_tgt = utils.get_concept_vec_filename(
                args.control_method, concept, rep_token, tgt_llm.model_name, use_soft_labels
            )
            if not os.path.isfile(vec_path_tgt):
                print(f"[skip target-native] no target directions {concept}", flush=True)
                for coef in coefs:
                    for pr in prompts:
                        target_native_cache[(concept, pr["text"], coef)] = None
            else:
                try:
                    tgt_steering_ctrl.load(
                        concept=concept,
                        rep_token=rep_token,
                        model_name=tgt_llm.model_name,
                        path=os.path.join(utils.DATA_DIR, "directions"),
                        load_concat_layers=False,
                        hidden_state="block",
                        use_soft_labels=use_soft_labels,
                        head_agg="max",
                    )
                except FileNotFoundError as e:
                    print(f"[skip target-native] {concept}: {e}", flush=True)
                    for coef in coefs:
                        for pr in prompts:
                            target_native_cache[(concept, pr["text"], coef)] = None
                else:
                    for coef in coefs:
                        for pr in prompts:
                            t = pr["text"]
                            target_native_cache[(concept, t, coef)] = tgt_steering_ctrl.generate(
                                t,
                                hidden_state="block",
                                control_coef=coef,
                                concat_layers_list=[],
                                layers_to_control=tgt_layers,
                                max_new_tokens=args.max_tokens,
                                do_sample=False,
                            )

            # --- Transfer (mapped from source) ---
            vec_path = utils.get_concept_vec_filename(
                args.control_method, concept, rep_token, args.source_model, use_soft_labels
            )
            wp = w_pkl_path(args.source_model, args.target_model, args.concept_type, concept)
            if not os.path.isfile(vec_path) or not wp.is_file():
                print(f"[skip transfer] {concept}", flush=True)
                continue
            with open(vec_path, "rb") as f:
                src_dirs = pickle.load(f)
            with open(wp, "rb") as f:
                W_layers = pickle.load(f)
            transferred = apply_transfer(W_layers, src_dirs)
            tgt_steering_ctrl.individual_directions = transferred
            tgt_steering_ctrl.get_all_directions([])
            for coef in coefs:
                for pr in prompts:
                    t = pr["text"]
                    transfer_cache[(concept, t, coef)] = tgt_steering_ctrl.generate(
                        t,
                        hidden_state="block",
                        control_coef=coef,
                        concat_layers_list=[],
                        layers_to_control=tgt_layers,
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                    )
        _unload(tgt_llm)
    else:
        _unload(tgt_llm)
        tgt_llm = None

    # --- Phase 2: source model — native steered ---
    print("[phase 2] Source model: native steered...", flush=True)
    src_llm = utils.select_llm(args.source_model)
    src_layers = layer_indices_steered(src_llm.language_model.config.num_hidden_layers)
    native_cache: dict[tuple[str, str, float], str | None] = {}

    for concept in concepts:
        vec_path = utils.get_concept_vec_filename(
            args.control_method, concept, rep_token, args.source_model, use_soft_labels
        )
        if not os.path.isfile(vec_path):
            print(f"[skip native] no directions {concept}", flush=True)
            for coef in coefs:
                for pr in prompts:
                    native_cache[(concept, pr["text"], coef)] = None
            continue
        native_ctrl = NeuralController(
            src_llm,
            src_llm.tokenizer,
            control_method=args.control_method,
            n_components=1,
            start_from_token=args.start_from_token,
        )
        try:
            native_ctrl.load(
                concept=concept,
                rep_token=rep_token,
                model_name=src_llm.model_name,
                path=os.path.join(utils.DATA_DIR, "directions"),
                load_concat_layers=False,
                hidden_state="block",
                use_soft_labels=use_soft_labels,
                head_agg="max",
            )
        except FileNotFoundError as e:
            print(f"[skip native] {concept}: {e}", flush=True)
            for coef in coefs:
                for pr in prompts:
                    native_cache[(concept, pr["text"], coef)] = None
            continue
        for coef in coefs:
            for pr in prompts:
                t = pr["text"]
                native_cache[(concept, t, coef)] = native_ctrl.generate(
                    t,
                    hidden_state="block",
                    control_coef=coef,
                    concat_layers_list=[],
                    layers_to_control=src_layers,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                )

    _unload(src_llm)

    # --- Phase 3: target-native then transfer (if not keep_target_loaded) ---
    if not args.keep_target_loaded:
        print("[phase 3a] Target model: target-native steered...", flush=True)
        tgt_llm = utils.select_llm(args.target_model)
        tgt_layers = layer_indices_steered(tgt_llm.language_model.config.num_hidden_layers)
        tgt_steering_ctrl = NeuralController(
            tgt_llm,
            tgt_llm.tokenizer,
            control_method=args.control_method,
            n_components=1,
            start_from_token=args.start_from_token,
        )
        for concept in concepts:
            vec_path_tgt = utils.get_concept_vec_filename(
                args.control_method, concept, rep_token, tgt_llm.model_name, use_soft_labels
            )
            if not os.path.isfile(vec_path_tgt):
                print(f"[skip target-native] no target directions {concept}", flush=True)
                for coef in coefs:
                    for pr in prompts:
                        target_native_cache[(concept, pr["text"], coef)] = None
                continue
            try:
                tgt_steering_ctrl.load(
                    concept=concept,
                    rep_token=rep_token,
                    model_name=tgt_llm.model_name,
                    path=os.path.join(utils.DATA_DIR, "directions"),
                    load_concat_layers=False,
                    hidden_state="block",
                    use_soft_labels=use_soft_labels,
                    head_agg="max",
                )
            except FileNotFoundError as e:
                print(f"[skip target-native] {concept}: {e}", flush=True)
                for coef in coefs:
                    for pr in prompts:
                        target_native_cache[(concept, pr["text"], coef)] = None
                continue
            for coef in coefs:
                for pr in prompts:
                    t = pr["text"]
                    target_native_cache[(concept, t, coef)] = tgt_steering_ctrl.generate(
                        t,
                        hidden_state="block",
                        control_coef=coef,
                        concat_layers_list=[],
                        layers_to_control=tgt_layers,
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                    )

        print("[phase 3b] Target model: transfer steered...", flush=True)
        for concept in concepts:
            vec_path = utils.get_concept_vec_filename(
                args.control_method, concept, rep_token, args.source_model, use_soft_labels
            )
            wp = w_pkl_path(args.source_model, args.target_model, args.concept_type, concept)
            if not os.path.isfile(vec_path) or not wp.is_file():
                continue
            with open(vec_path, "rb") as f:
                src_dirs = pickle.load(f)
            with open(wp, "rb") as f:
                W_layers = pickle.load(f)
            transferred = apply_transfer(W_layers, src_dirs)
            tgt_steering_ctrl.individual_directions = transferred
            tgt_steering_ctrl.get_all_directions([])
            for coef in coefs:
                for pr in prompts:
                    t = pr["text"]
                    transfer_cache[(concept, t, coef)] = tgt_steering_ctrl.generate(
                        t,
                        hidden_state="block",
                        control_coef=coef,
                        concat_layers_list=[],
                        layers_to_control=tgt_layers,
                        max_new_tokens=args.max_tokens,
                        do_sample=False,
                    )
        _unload(tgt_llm)

    # --- Write JSONL ---
    n_rows = 0
    for concept in concepts:
        for pr in prompts:
            t = pr["text"]
            bl = baseline_by_text.get(t)
            for coef in coefs:
                nv = native_cache.get((concept, t, coef))
                tr = transfer_cache.get((concept, t, coef))
                tnv = target_native_cache.get((concept, t, coef))
                if bl is None or all(x is None for x in (nv, tr, tnv)):
                    continue
                row = {
                    "concept_type": args.concept_type,
                    "concept": concept,
                    "prompt_id": pr["id"],
                    "version": pr["eval_version"],
                    "eval_version": pr["eval_version"],
                    "prompt": t,
                    "coef": coef,
                    "rep_token": rep_token,
                    "source_model": args.source_model,
                    "target_model": args.target_model,
                    "baseline": bl,
                    "native_source_steered": nv,
                    "native_target_steered": tnv,
                    "transfer_target_steered": tr,
                }
                with open(out_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"Done. Wrote {n_rows} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
