from __future__ import annotations

from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd

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

    rows = []
    best_alpha = None
    best_score = -np.inf
    for alpha in np.linspace(0.0, 1.0, 21):
        scores = alpha * perch_scores + (1.0 - alpha) * passt_scores
        score = challenge_score_from_arrays(perch_targets, scores, label_space.labels)
        rows.append({"perch_alpha": alpha, "val_challenge_score": score})
        if score > best_score:
            best_score = score
            best_alpha = alpha

    results = pd.DataFrame(rows)
    print("Ensemble alpha sweep")
    print(results.to_string(index=False, formatters={"perch_alpha": "{:.2f}".format, "val_challenge_score": "{:.5f}".format}))
    print()
    print(f"best perch_alpha: {best_alpha:.2f}")
    print(f"best val_challenge_score: {best_score:.5f}")

    best_scores = best_alpha * perch_scores + (1.0 - best_alpha) * passt_scores
    output_path = cfg.get("ensemble_val_predictions_path", "")
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, targets=perch_targets, scores=best_scores, perch_alpha=np.asarray(best_alpha))
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
