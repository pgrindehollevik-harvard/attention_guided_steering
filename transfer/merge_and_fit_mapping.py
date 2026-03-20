#!/usr/bin/env python3
"""
Fit per-layer linear maps X such that A_tgt ≈ A_src @ X (row = one prompt).
Transferred direction: v_tgt = normalize(X.T @ v_src) for column vectors v.

Example:
  python transfer/merge_and_fit_mapping.py \\
    --src_npz data/paired_activations/acts_llama_3.1_70b_fears_fire.npz \\
    --tgt_npz data/paired_activations/acts_llama_3.3_70b_fears_fire.npz \\
    --source_model llama_3.1_70b --target_model llama_3.3_70b \\
    -c fears --concept fire --ridge 1e-2
"""
from __future__ import annotations

from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import pickle

import numpy as np

from transfer.transfer_utils import default_mapping_path, ensure_repo_cwd, transfer_map_dir


def parse_args():
    p = argparse.ArgumentParser(description="Fit ridge maps between paired activation files.")
    p.add_argument("--src_npz", required=True, help="Source model activations .npz")
    p.add_argument("--tgt_npz", required=True, help="Target model activations .npz")
    p.add_argument("--source_model", required=True)
    p.add_argument("--target_model", required=True)
    p.add_argument("--concept_type", "-c", required=True)
    p.add_argument("--concept", required=True)
    p.add_argument(
        "--ridge",
        type=float,
        default=1e-2,
        help="Ridge λ on A_src^T A_src (stabilizes d×d solve).",
    )
    p.add_argument("--out", default=None, help="Output .npz for W layers dict + meta")
    return p.parse_args()


def fit_layer_map(A_src: np.ndarray, A_tgt: np.ndarray, lam: float) -> np.ndarray:
    """
    Rows = prompts. Solve A_src @ X = A_tgt, X is (d, d).
    Ridge: X = (A_src^T A_src + λI)^{-1} A_src^T A_tgt
    """
    d = A_src.shape[1]
    AtA = A_src.T @ A_src
    rhs = A_src.T @ A_tgt
    X = np.linalg.solve(AtA + lam * np.eye(d), rhs)
    return X.astype(np.float32)


def main():
    ensure_repo_cwd()
    args = parse_args()

    src = np.load(args.src_npz, allow_pickle=True)
    tgt = np.load(args.tgt_npz, allow_pickle=True)

    A_s = np.asarray(src["activations"], dtype=np.float64)
    A_t = np.asarray(tgt["activations"], dtype=np.float64)
    if A_s.shape != A_t.shape:
        raise ValueError(f"Shape mismatch: src {A_s.shape} vs tgt {A_t.shape}")

    n, L, d = A_s.shape
    layer_indices = np.asarray(src["layer_indices"])
    if not np.array_equal(layer_indices, np.asarray(tgt["layer_indices"])):
        raise ValueError("layer_indices differ between src and tgt files.")

    W_layers: dict[int, np.ndarray] = {}
    for j in range(L):
        X = fit_layer_map(A_s[:, j, :], A_t[:, j, :], args.ridge)
        li = int(layer_indices[j])
        W_layers[li] = X

    out_path = args.out
    if out_path is None:
        out_path = str(
            default_mapping_path(
                args.source_model, args.target_model, args.concept_type, args.concept
            )
        )

    transfer_map_dir()
    with open(out_path.replace(".npz", "_W.pkl"), "wb") as f:
        pickle.dump(W_layers, f)

    np.savez_compressed(
        out_path,
        source_model=args.source_model,
        target_model=args.target_model,
        concept_type=args.concept_type,
        concept=args.concept,
        ridge=args.ridge,
        layer_indices=layer_indices,
        src_npz=np.array(args.src_npz),
        tgt_npz=np.array(args.tgt_npz),
        n_prompts=n,
        hidden_size=d,
    )
    print(f"Saved W dict to {out_path.replace('.npz', '_W.pkl')} and meta to {out_path}")


if __name__ == "__main__":
    main()
