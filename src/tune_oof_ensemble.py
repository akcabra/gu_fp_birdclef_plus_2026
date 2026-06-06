from __future__ import annotations

from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import build_label_space, load_tables
from src.ensemble_predictions import (
    alpha_grid,
    blend_scores,
    global_alpha_sweep,
    load_prediction_file,
    tune_classwise_alphas,
)
from src.metrics import challenge_score_from_arrays


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def prediction_dir_from_config(path_value: str, model_name: str) -> Path:
    if not path_value:
        raise ValueError(
            f"{model_name}_val_predictions_path is empty. Set it to either the CV prediction directory "
            f"or one fold prediction file inside that directory."
        )

    path = Path(path_value)
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    if path.suffix == ".npz":
        return path.parent
    return path


def fold_prediction_paths(prediction_dir: Path, model_name: str) -> dict[int, Path]:
    paths = {}
    for path in sorted(prediction_dir.glob(f"fold*_{model_name}_val_predictions.npz")):
        stem = path.stem
        fold_text = stem.split("_", 1)[0].replace("fold", "")
        if fold_text.isdigit():
            paths[int(fold_text)] = path
    if not paths:
        raise FileNotFoundError(
            f"No fold prediction files matching fold*_{model_name}_val_predictions.npz in {prediction_dir}"
        )
    return paths


def validate_fold_pair(fold: int, perch_data: dict[str, np.ndarray], passt_data: dict[str, np.ndarray]) -> None:
    if perch_data["scores"].shape != passt_data["scores"].shape:
        raise ValueError(
            f"Fold {fold} prediction shapes differ: "
            f"Perch {perch_data['scores'].shape}, PaSST {passt_data['scores'].shape}"
        )
    if not np.array_equal(perch_data["targets"], passt_data["targets"]):
        raise ValueError(f"Fold {fold} Perch and PaSST targets differ")
    if "source_ids" in perch_data and "source_ids" in passt_data:
        if not np.array_equal(perch_data["source_ids"].astype(str), passt_data["source_ids"].astype(str)):
            raise ValueError(f"Fold {fold} Perch and PaSST source_ids differ")


def load_oof_predictions(perch_dir: Path, passt_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    perch_paths = fold_prediction_paths(perch_dir, "perch")
    passt_paths = fold_prediction_paths(passt_dir, "passt")
    folds = sorted(set(perch_paths) & set(passt_paths))
    missing_perch = sorted(set(passt_paths) - set(perch_paths))
    missing_passt = sorted(set(perch_paths) - set(passt_paths))
    if missing_perch or missing_passt:
        raise ValueError(
            "Perch and PaSST fold prediction sets do not match. "
            f"Missing Perch folds: {missing_perch}; missing PaSST folds: {missing_passt}"
        )
    if not folds:
        raise ValueError("No shared prediction folds found")

    targets = []
    perch_scores = []
    passt_scores = []
    source_ids = []
    for fold in folds:
        perch_data = load_prediction_file(perch_paths[fold])
        passt_data = load_prediction_file(passt_paths[fold])
        validate_fold_pair(fold, perch_data, passt_data)
        targets.append(perch_data["targets"])
        perch_scores.append(perch_data["scores"])
        passt_scores.append(passt_data["scores"])
        if "source_ids" in perch_data:
            source_ids.append(perch_data["source_ids"].astype(str))
        print(
            f"Loaded fold {fold}: rows={len(perch_data['targets'])}, "
            f"Perch={perch_paths[fold]}, PaSST={passt_paths[fold]}",
            flush=True,
        )

    source_id_array = np.concatenate(source_ids, axis=0) if source_ids else np.asarray([], dtype=str)
    return (
        np.concatenate(targets, axis=0).astype(np.float32),
        np.concatenate(perch_scores, axis=0).astype(np.float32),
        np.concatenate(passt_scores, axis=0).astype(np.float32),
        source_id_array,
    )


def alpha_sweep_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix:
        return path.with_name(f"{path.stem}_alpha_sweep.csv")
    return path / "ensemble_alpha_sweep.csv"


def main() -> None:
    start = perf_counter()
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    _, _, _, sample_submission = load_tables(cfg["data_root"])
    label_space = build_label_space(sample_submission)

    perch_dir = prediction_dir_from_config(cfg.get("perch_val_predictions_path", ""), "perch")
    passt_dir = prediction_dir_from_config(cfg.get("passt_val_predictions_path", ""), "passt")
    print("OOF ensemble prediction directories")
    print(f"Perch: {perch_dir}")
    print(f"PaSST: {passt_dir}")
    print()

    y_true, perch_scores, passt_scores, source_ids = load_oof_predictions(perch_dir, passt_dir)
    print()
    print(f"OOF rows: {len(y_true)}")
    print(f"Prediction shape: {perch_scores.shape}")
    print()

    grid = alpha_grid(cfg)
    results, best_alpha, best_score, best_scores = global_alpha_sweep(
        y_true,
        perch_scores,
        passt_scores,
        label_space.labels,
        grid,
    )
    print("OOF ensemble alpha sweep")
    print(results.to_string(index=False, formatters={"perch_alpha": "{:.2f}".format, "val_challenge_score": "{:.5f}".format}))
    print()
    print(f"best global perch_alpha: {best_alpha:.2f}")
    print(f"best global OOF challenge score: {best_score:.5f}")

    blend_mode = str(cfg.get("ensemble_blend_mode", "global")).lower()
    if blend_mode not in {"global", "classwise"}:
        raise ValueError("ensemble_blend_mode must be 'global' or 'classwise'")

    classwise_alphas = None
    if blend_mode == "classwise":
        shrinkage = float(cfg.get("ensemble_classwise_shrinkage", 0.5))
        min_positives = int(cfg.get("ensemble_classwise_min_positives", 3))
        classwise_alphas, classwise_table = tune_classwise_alphas(
            y_true,
            perch_scores,
            passt_scores,
            label_space.labels,
            grid,
            best_alpha,
            shrinkage,
            min_positives,
        )
        best_scores = blend_scores(perch_scores, passt_scores, classwise_alphas[np.newaxis, :])
        classwise_score = challenge_score_from_arrays(y_true, best_scores, label_space.labels)
        print()
        print("Class-wise OOF blending")
        print(f"global_anchor_perch_alpha: {best_alpha:.2f}")
        print(f"shrinkage: {shrinkage:.2f}")
        print(f"min positives for tuning: {min_positives}")
        print(f"class-wise OOF challenge score: {classwise_score:.5f}")
        print(f"delta vs best global: {classwise_score - best_score:+.5f}")
        print(
            classwise_table["perch_alpha"]
            .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            .to_string(float_format="{:.3f}".format)
        )

        alpha_path = cfg.get("ensemble_classwise_alphas_path", "")
        if alpha_path:
            alpha_path = Path(alpha_path)
            alpha_path.parent.mkdir(parents=True, exist_ok=True)
            classwise_table.to_csv(alpha_path, index=False)
            print(f"Saved class-wise alphas to {alpha_path}")

    output_path = cfg.get("ensemble_val_predictions_path", "")
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_arrays = {
            "targets": y_true,
            "scores": best_scores,
            "perch_scores": perch_scores,
            "passt_scores": passt_scores,
            "perch_alpha": np.asarray(best_alpha),
            "blend_mode": np.asarray(blend_mode),
            "split_eval": np.asarray("cv_oof"),
        }
        if len(source_ids):
            save_arrays["source_ids"] = source_ids
        if classwise_alphas is not None:
            save_arrays["classwise_perch_alpha"] = classwise_alphas
        np.savez_compressed(output_path, **save_arrays)
        print(f"Saved OOF blended predictions to {output_path}")

        sweep_path = alpha_sweep_path(output_path)
        results.to_csv(sweep_path, index=False)
        print(f"Saved alpha sweep to {sweep_path}")

    print(f"Total run time: {format_duration(perf_counter() - start)}")


if __name__ == "__main__":
    main()
