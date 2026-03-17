"""
Offline training script for the DVS multivariate regression calibration.

This script:
  - loads the dataset and metadata created by `hardware.dvs_regression_dataset`
  - standardizes the camnorm inputs (b1, s1, b2, s2)
  - builds the polynomial feature vector described in `docs/dvs_full_cal.md`
  - solves four independent least-squares problems for (X, Y, ax, ay)
  - saves a JSON model artifact into `perception/calibration_files/`

Usage (from repo root, after dataset collection):
    .venv/bin/python -m hardware.dvs_regression_training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


DEFAULT_DATASET_PATH = Path("perception/calibration_files/dvs_pose_dataset.npz")
DEFAULT_META_PATH = Path("perception/calibration_files/dvs_pose_dataset_meta.json")
DEFAULT_MODEL_PATH = Path("perception/calibration_files/dvs_pose_regression_model.json")


EXPECTED_COLUMNS = [
    "b1",
    "s1",
    "b2",
    "s2",
    "X_true",
    "Y_true",
    "ax_true",
    "ay_true",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train multivariate linear regression model "
            "f(b1, s1, b2, s2) -> (X, Y, ax, ay) "
            "from the DVS regression dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="Path to dvs_pose_dataset.npz produced by hardware.dvs_regression_dataset",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=str(DEFAULT_META_PATH),
        help="Path to dvs_pose_dataset_meta.json sidecar metadata",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Output path for trained model JSON",
    )
    return parser.parse_args()


def load_dataset(
    dataset_path: Path, meta_path: Path
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    npz = np.load(dataset_path)
    if "data" not in npz:
        raise KeyError(
            f"Expected key 'data' in {dataset_path}, found keys: {list(npz.files)}"
        )
    data = np.asarray(npz["data"], dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(
            f"Expected dataset of shape (N, 8); got {data.shape!r} from {dataset_path}"
        )

    with meta_path.open("r") as f:
        meta = json.load(f)

    cols = meta.get("column_names")
    if cols is None:
        raise KeyError(f"Metadata {meta_path} missing 'column_names'")
    if list(cols) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected column_names in {meta_path}: {cols!r} "
            f"(expected {EXPECTED_COLUMNS!r})"
        )

    return data, meta


def compute_standardization_stats(inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std for camnorm inputs.

    Guard against zero std by replacing any ~zero std with 1.0 so that
    standardization is well-defined. This also matches the expectation that
    runtime code should never divide by zero.
    """
    mean = inputs.mean(axis=0)
    std = inputs.std(axis=0, ddof=0)

    std_safe = std.copy()
    std_safe[std_safe < 1e-8] = 1.0
    return mean, std_safe


def build_design_matrix(
    inputs_z: np.ndarray,
) -> np.ndarray:
    """
    Build design matrix M where each row is:
        [1, b1_z, b2_z, s1_z, s2_z,
         b1_z*s1_z, b1_z*s2_z, b2_z*s1_z, b2_z*s2_z]
    following docs/dvs_full_cal.md.
    """
    if inputs_z.shape[1] != 4:
        raise ValueError(f"Expected 4 standardized inputs per row, got {inputs_z.shape[1]}")

    b1_z = inputs_z[:, 0]
    s1_z = inputs_z[:, 1]
    b2_z = inputs_z[:, 2]
    s2_z = inputs_z[:, 3]

    ones = np.ones_like(b1_z)

    features = np.column_stack(
        [
            ones,
            b1_z,
            b2_z,
            s1_z,
            s2_z,
            b1_z * s1_z,
            b1_z * s2_z,
            b2_z * s1_z,
            b2_z * s2_z,
        ]
    )
    return features


def fit_linear_regression(
    M: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """
    Solve least squares M @ W.T ≈ targets, where:
      - M has shape (N, 9)
      - targets has shape (N, 4) with columns [X, Y, ax, ay]
      - return W with shape (4, 9), matching docs/dvs_full_cal.md
    """
    if M.ndim != 2 or M.shape[1] != 9:
        raise ValueError(f"Design matrix must have shape (N, 9), got {M.shape}")
    if targets.ndim != 2 or targets.shape[0] != M.shape[0] or targets.shape[1] != 4:
        raise ValueError(
            f"Targets must have shape (N, 4) with same N as M; "
            f"got M={M.shape}, targets={targets.shape}"
        )

    # Solve four independent least-squares problems.
    # Using rcond=None for future-proof numpy behaviour.
    W_rows: List[np.ndarray] = []
    for i in range(4):
        coeffs, *_ = np.linalg.lstsq(M, targets[:, i], rcond=None)
        W_rows.append(coeffs.astype(float))

    W = np.vstack(W_rows)
    return W


def build_model_artifact(
    input_mean: np.ndarray,
    input_std: np.ndarray,
    W: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build JSON-serializable model artifact following the spec in dvs_full_cal.md.
    """
    if W.shape != (4, 9):
        raise ValueError(f"Expected W shape (4, 9), got {W.shape}")

    model: Dict[str, Any] = {
        "model_type": "linear_regression",
        "input_order": ["b1", "s1", "b2", "s2"],
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "feature_order": [
            "1",
            "b1_z",
            "b2_z",
            "s1_z",
            "s2_z",
            "b1_z*s1_z",
            "b1_z*s2_z",
            "b2_z*s1_z",
            "b2_z*s2_z",
        ],
        "output_order": ["X", "Y", "ax", "ay"],
        "W": W.tolist(),
        "units": {"X": "m", "Y": "m", "ax": "rad", "ay": "rad"},
        "dataset": {
            "grid_shape": meta.get("grid_shape"),
            "tilt_angles_deg": meta.get("tilt_angles_deg"),
            "total_samples": meta.get("total_samples"),
            "frames_per_pose_target": meta.get("frames_per_pose_target"),
        },
    }
    return model


def train_and_save(
    dataset_path: Path,
    meta_path: Path,
    output_model_path: Path,
) -> None:
    print(f"Loading dataset from {dataset_path} and metadata from {meta_path}...")
    data, meta = load_dataset(dataset_path, meta_path)

    # Split inputs and targets
    inputs = data[:, 0:4]  # [b1, s1, b2, s2]
    targets = data[:, 4:8]  # [X_true, Y_true, ax_true, ay_true]

    print(f"Dataset loaded: {data.shape[0]} samples.")

    # Standardization stats and standardized inputs
    input_mean, input_std = compute_standardization_stats(inputs)
    inputs_z = (inputs - input_mean) / input_std

    # Design matrix and regression
    M = build_design_matrix(inputs_z)
    W = fit_linear_regression(M, targets)

    # Build artifact and save
    model = build_model_artifact(input_mean, input_std, W, meta)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with output_model_path.open("w") as f:
        json.dump(model, f, indent=2)

    print(f"Saved trained regression model to {output_model_path}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    meta_path = Path(args.meta)
    output_model_path = Path(args.output_model)

    train_and_save(dataset_path, meta_path, output_model_path)


if __name__ == "__main__":
    main()

