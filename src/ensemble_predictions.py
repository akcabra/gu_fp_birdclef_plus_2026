from __future__ import annotations

from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import attach_targets, build_label_space, load_tables, make_mixed_split
from src.evaluation import print_validation_group_summary, validation_group_summary
from src.metrics import challenge_score_from_arrays


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def load_prediction_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["targets"].astype(np.float32), data["scores"].astype(np.float32)


def alpha_grid(cfg: dict) -> np.ndarray:
    values = cfg.get("ensemble_alpha_values")
    if values is not None:
        grid = np.asarray(values, dtype=np.float32)
    else:
        step = float(cfg.get("ensemble_alpha_step", 0.05))
        if step <= 0.0 or step > 1.0:
            raise ValueError("ensemble_alpha_step must be in (0, 1]")
        grid = np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float32)
    grid = np.unique(np.clip(grid, 0.0, 1.0))
    if len(grid) == 0:
        raise ValueError("ensemble alpha grid is empty")
    return grid


def blend_scores(perch_scores: np.ndarray, passt_scores: np.ndarray, perch_alpha: float | np.ndarray) -> np.ndarray:
    return perch_alpha * perch_scores + (1.0 - perch_alpha) * passt_scores


def global_alpha_sweep(
    y_true: np.ndarray,
    perch_scores: np.ndarray,
    passt_scores: np.ndarray,
    labels: list[str],
    grid: np.ndarray,
) -> tuple[pd.DataFrame, float, float, np.ndarray]:
    rows = []
    best_alpha = float(grid[0])
    best_score = -np.inf
    for alpha in grid:
        scores = blend_scores(perch_scores, passt_scores, float(alpha))
        score = challenge_score_from_arrays(y_true, scores, labels)
        rows.append({"perch_alpha": float(alpha), "val_challenge_score": score})
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    best_scores = blend_scores(perch_scores, passt_scores, best_alpha)
    return pd.DataFrame(rows), best_alpha, float(best_score), best_scores


def tune_classwise_alphas(
    y_true: np.ndarray,
    perch_scores: np.ndarray,
    passt_scores: np.ndarray,
    labels: list[str],
    grid: np.ndarray,
    global_alpha: float,
    shrinkage: float,
    min_positives: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("ensemble_classwise_shrinkage must be in [0, 1]")
    if min_positives < 1:
        raise ValueError("ensemble_classwise_min_positives must be >= 1")

    class_alphas = np.full(len(labels), global_alpha, dtype=np.float32)
    rows = []
    for idx, label in enumerate(labels):
        y_col = y_true[:, idx]
        positives = int(y_col.sum())
        negatives = int(len(y_col) - positives)
        if positives < min_positives or negatives == 0:
            rows.append(
                {
                    "label": label,
                    "positives": positives,
                    "negatives": negatives,
                    "best_alpha_raw": global_alpha,
                    "perch_alpha": global_alpha,
                    "best_auc": np.nan,
                    "tuned": False,
                }
            )
            continue

        best_alpha = global_alpha
        best_auc = -np.inf
        for alpha in grid:
            scores = blend_scores(perch_scores[:, idx], passt_scores[:, idx], float(alpha))
            auc = roc_auc_score(y_col, scores)
            if auc > best_auc:
                best_auc = float(auc)
                best_alpha = float(alpha)

        shrunk_alpha = global_alpha + shrinkage * (best_alpha - global_alpha)
        class_alphas[idx] = np.float32(shrunk_alpha)
        rows.append(
            {
                "label": label,
                "positives": positives,
                "negatives": negatives,
                "best_alpha_raw": best_alpha,
                "perch_alpha": float(shrunk_alpha),
                "best_auc": best_auc,
                "tuned": True,
            }
        )

    return class_alphas, pd.DataFrame(rows)


def main() -> None:
    start = perf_counter()
    cfg = load_config(PROJECT_ROOT / "config.yaml")

    train_csv, soundscape_csv, taxonomy, sample_submission = load_tables(cfg["data_root"])
    label_space = build_label_space(sample_submission)
    train_rows, val_rows = make_mixed_split(
        train_csv,
        soundscape_csv,
        cfg["data_root"],
        cfg["soundscape_val_fraction"],
        cfg["seed"],
    )
    train_rows = attach_targets(train_rows, label_space)

    perch_targets, perch_scores = load_prediction_file(cfg["perch_val_predictions_path"])
    passt_targets, passt_scores = load_prediction_file(cfg["passt_val_predictions_path"])
    if perch_scores.shape != passt_scores.shape:
        raise ValueError(f"Prediction shapes differ: Perch {perch_scores.shape}, PaSST {passt_scores.shape}")
    if not np.array_equal(perch_targets, passt_targets):
        raise ValueError("Perch and PaSST prediction files have different validation targets/order")

    grid = alpha_grid(cfg)
    results, best_alpha, best_score, best_scores = global_alpha_sweep(
        perch_targets,
        perch_scores,
        passt_scores,
        label_space.labels,
        grid,
    )
    print("Ensemble alpha sweep")
    print(results.to_string(index=False, formatters={"perch_alpha": "{:.2f}".format, "val_challenge_score": "{:.5f}".format}))
    print()
    print(f"best perch_alpha: {best_alpha:.2f}")
    print(f"best val_challenge_score: {best_score:.5f}")

    blend_mode = str(cfg.get("ensemble_blend_mode", "global")).lower()
    classwise_alphas = None
    classwise_table = None
    if blend_mode not in {"global", "classwise"}:
        raise ValueError("ensemble_blend_mode must be 'global' or 'classwise'")

    if blend_mode == "classwise":
        shrinkage = float(cfg.get("ensemble_classwise_shrinkage", 0.5))
        min_positives = int(cfg.get("ensemble_classwise_min_positives", 3))
        classwise_alphas, classwise_table = tune_classwise_alphas(
            perch_targets,
            perch_scores,
            passt_scores,
            label_space.labels,
            grid,
            best_alpha,
            shrinkage,
            min_positives,
        )
        best_scores = blend_scores(perch_scores, passt_scores, classwise_alphas[np.newaxis, :])
        classwise_score = challenge_score_from_arrays(perch_targets, best_scores, label_space.labels)

        print()
        print("Class-wise blending")
        print(f"global_anchor_perch_alpha: {best_alpha:.2f}")
        print(f"shrinkage: {shrinkage:.2f}")
        print(f"min positives for tuning: {min_positives}")
        print(f"class-wise val_challenge_score: {classwise_score:.5f}")
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
            "targets": perch_targets,
            "scores": best_scores,
            "perch_alpha": np.asarray(best_alpha),
            "blend_mode": np.asarray(blend_mode),
        }
        if classwise_alphas is not None:
            save_arrays["classwise_perch_alpha"] = classwise_alphas
        np.savez_compressed(output_path, **save_arrays)
        print(f"Saved blended validation predictions to {output_path}")

    perch_mapping = pd.read_csv(cfg["perch_label_mapping_path"])
    soundscape_train_rows = train_rows[train_rows["source"] == "soundscape"]
    summary = validation_group_summary(
        perch_targets,
        best_scores,
        label_space.labels,
        taxonomy,
        perch_mapping,
        train_rows,
        soundscape_train_rows,
    )
    print_validation_group_summary(summary)
    print(f"Total run time: {format_duration(perf_counter() - start)}")


if __name__ == "__main__":
    main()
