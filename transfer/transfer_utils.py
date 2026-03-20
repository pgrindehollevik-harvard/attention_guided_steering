"""
Shared paths and helpers for activation collection and transfer mapping.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_repo_cwd() -> Path:
    """Run scripts from repo root so data/ paths resolve."""
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return REPO_ROOT


def paired_act_dir() -> Path:
    p = REPO_ROOT / "data" / "paired_activations"
    p.mkdir(parents=True, exist_ok=True)
    return p


def transfer_map_dir() -> Path:
    p = REPO_ROOT / "data" / "transfer_mappings"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_act_path(model_name: str, concept_type: str, concept_slug: str) -> Path:
    safe = concept_slug.replace("/", "_").replace(" ", "_")[:120]
    return paired_act_dir() / f"acts_{model_name}_{concept_type}_{safe}.npz"


def default_mapping_path(source_model: str, target_model: str, concept_type: str, concept_slug: str) -> Path:
    safe = concept_slug.replace("/", "_").replace(" ", "_")[:120]
    return transfer_map_dir() / f"W_{source_model}_to_{target_model}_{concept_type}_{safe}.npz"


def w_pkl_path(source_model: str, target_model: str, concept_type: str, concept_slug: str) -> Path:
    """Pickle of layer -> W matrix; matches merge_and_fit_mapping.py default output."""
    p = default_mapping_path(source_model, target_model, concept_type, concept_slug)
    return p.with_name(p.stem + "_W.pkl")


def layer_indices_steered(num_hidden_layers: int) -> list[int]:
    """Match 2_steer.py / NeuralController: layers 1 .. num_hidden_layers-1."""
    return list(range(1, num_hidden_layers))


def save_run_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_run_meta(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
