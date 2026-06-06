from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import build_label_space, load_tables, make_split_from_plan
from src.evaluation import labels_in_rows
from src.metrics import per_class_auc


def scalar_string(arrays: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in arrays:
        return None
    value = arrays[key]
    if value.shape == ():
        return str(value.item())
    return str(value.reshape(-1)[0])


def scalar_int(arrays: np.lib.npyio.NpzFile, key: str) -> int | None:
    value = scalar_string(arrays, key)
    if value is None:
        return None
    return int(value)


def apply_prediction_split_metadata(cfg: dict, predictions: np.lib.npyio.NpzFile) -> dict:
    cfg = dict(cfg)
    split_eval = scalar_string(predictions, "split_eval")
    split_fold = scalar_int(predictions, "split_fold")
    split_plan_path = scalar_string(predictions, "split_plan_path")

    if split_eval:
        cfg["soundscape_eval_split"] = split_eval
    if split_fold is not None and split_fold >= 0:
        cfg["soundscape_fold"] = split_fold
    if split_plan_path:
        cfg["soundscape_split_plan_path"] = split_plan_path
    return cfg


def resolve_prediction_paths(paths: list[Path]) -> list[Path]:
    resolved = []
    for path in paths:
        if path.is_dir():
            direct = sorted(path.glob("*.npz"))
            resolved.extend(direct if direct else sorted(path.rglob("*.npz")))
        else:
            resolved.append(path)
    resolved = [path for path in resolved if path.suffix == ".npz"]
    if not resolved:
        raise ValueError("No .npz prediction files found")
    return resolved


def train_label_counts(train_rows: pd.DataFrame, labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for row_labels in train_rows["labels"]:
        for label in row_labels:
            label = str(label)
            if label in counts:
                counts[label] += 1
    return counts


def build_per_class_breakdown(
    y_true: np.ndarray,
    scores: np.ndarray,
    labels: list[str],
    taxonomy: pd.DataFrame,
    perch_mapping: pd.DataFrame,
    train_rows_by_split: list[pd.DataFrame],
    soundscape_train_rows_by_split: list[pd.DataFrame],
) -> pd.DataFrame:
    per_class = per_class_auc(y_true, scores, labels)
    per_class["scored"] = per_class["auc"].notna()

    taxonomy_cols = [col for col in taxonomy.columns if col != "primary_label"]
    out = per_class.merge(
        taxonomy.assign(primary_label=taxonomy["primary_label"].astype(str)),
        left_on="label",
        right_on="primary_label",
        how="left",
    )

    matched_labels = set(perch_mapping.loc[perch_mapping["perch_index"].notna(), "primary_label"].astype(str))
    train_counts_by_split = [train_label_counts(train_rows, labels) for train_rows in train_rows_by_split]
    soundscape_labels_by_split = [labels_in_rows(rows) for rows in soundscape_train_rows_by_split]

    train_count_mean = {}
    train_count_min = {}
    train_count_max = {}
    train_soundscape_folds_present = {}
    for label in labels:
        counts = [split_counts[label] for split_counts in train_counts_by_split]
        train_count_mean[label] = float(np.mean(counts))
        train_count_min[label] = int(np.min(counts))
        train_count_max[label] = int(np.max(counts))
        train_soundscape_folds_present[label] = sum(label in split_labels for split_labels in soundscape_labels_by_split)

    out["train_count_mean"] = out["label"].map(train_count_mean).fillna(0.0)
    out["train_count_min"] = out["label"].map(train_count_min).fillna(0).astype(int)
    out["train_count_max"] = out["label"].map(train_count_max).fillna(0).astype(int)
    out["matched_perch_label"] = out["label"].isin(matched_labels)
    out["train_soundscape_folds_present"] = out["label"].map(train_soundscape_folds_present).fillna(0).astype(int)
    out["present_in_labeled_train_soundscapes"] = out["train_soundscape_folds_present"] > 0
    out["rare_train_count_le_5"] = (out["train_count_mean"] > 0) & (out["train_count_mean"] <= 5)

    ordered_cols = [
        "label",
        "auc",
        "positives",
        "negatives",
        "scored",
        "train_count_mean",
        "train_count_min",
        "train_count_max",
        "matched_perch_label",
        "present_in_labeled_train_soundscapes",
        "train_soundscape_folds_present",
        "rare_train_count_le_5",
        *taxonomy_cols,
    ]
    ordered_cols = [col for col in ordered_cols if col in out.columns]
    return out[ordered_cols].sort_values(["scored", "auc", "label"], ascending=[False, False, True])


def group_summary_from_per_class(
    per_class: pd.DataFrame,
    labels: list[str],
    taxonomy: pd.DataFrame,
    perch_mapping: pd.DataFrame,
) -> pd.DataFrame:
    taxonomy_by_label = taxonomy.set_index(taxonomy["primary_label"].astype(str))
    matched_labels = set(perch_mapping.loc[perch_mapping["perch_index"].notna(), "primary_label"].astype(str))
    rare_labels = set(per_class.loc[per_class["rare_train_count_le_5"], "label"].astype(str))
    train_soundscape_labels = set(
        per_class.loc[per_class["present_in_labeled_train_soundscapes"], "label"].astype(str)
    )

    groups = [
        ("all classes", set(labels)),
        ("matched Perch-label classes", matched_labels),
        ("unmatched Perch-label classes", set(labels) - matched_labels),
        ("rare classes train_count<=5", rare_labels),
        ("present in labeled train_soundscapes", train_soundscape_labels),
        ("absent from train_soundscapes", set(labels) - train_soundscape_labels),
    ]
    for class_name in ["Aves", "Insecta", "Amphibia", "Mammalia", "Reptilia"]:
        class_labels = set(taxonomy_by_label.loc[taxonomy_by_label["class_name"] == class_name].index.astype(str))
        groups.append((class_name, class_labels))

    rows = []
    for group_name, group_labels in groups:
        group = per_class[per_class["label"].isin(group_labels)]
        scored = group[group["auc"].notna()]
        rows.append(
            {
                "group": group_name,
                "labels": len(group),
                "scored_labels": len(scored),
                "val_positives": int(group["positives"].sum()),
                "mean_auc": scored["auc"].mean(),
                "median_auc": scored["auc"].median(),
            }
        )
    return pd.DataFrame(rows)


def output_stem(input_paths: list[Path], prediction_paths: list[Path]) -> str:
    if len(input_paths) == 1 and input_paths[0].is_dir():
        return f"{input_paths[0].name}_oof"
    if len(prediction_paths) == 1:
        return prediction_paths[0].stem
    parent_names = {path.parent.name for path in prediction_paths}
    if len(parent_names) == 1:
        return f"{prediction_paths[0].parent.name}_oof"
    return "validation_predictions_oof"


def default_output_paths(input_paths: list[Path], prediction_paths: list[Path], output_dir: Path | None) -> tuple[Path, Path]:
    if output_dir is None:
        output_dir = input_paths[0] if len(input_paths) == 1 and input_paths[0].is_dir() else prediction_paths[0].parent
    stem = output_stem(input_paths, prediction_paths)
    return output_dir / f"{stem}_per_class_auc.csv", output_dir / f"{stem}_group_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct per-species AUC and validation group summaries from saved validation predictions."
    )
    parser.add_argument(
        "predictions",
        type=Path,
        nargs="+",
        help="One or more prediction .npz files, or a directory containing CV fold prediction .npz files.",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml", help="Config used for data paths")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs. Defaults to the predictions directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    prediction_paths = resolve_prediction_paths(args.predictions)

    train_csv, soundscape_csv, taxonomy, sample_submission = load_tables(cfg["data_root"])
    label_space = build_label_space(sample_submission)

    split_targets = []
    split_scores = []
    train_rows_by_split = []
    soundscape_train_rows_by_split = []
    split_descriptions = []

    for predictions_path in prediction_paths:
        with np.load(predictions_path, allow_pickle=False) as predictions:
            split_cfg = apply_prediction_split_metadata(cfg, predictions)
            y_true_part = predictions["targets"]
            scores_part = predictions["scores"]

        if y_true_part.shape[1] != len(label_space.labels) or scores_part.shape[1] != len(label_space.labels):
            raise ValueError(
                "Prediction arrays do not match the configured label space: "
                f"path={predictions_path}, targets={y_true_part.shape}, "
                f"scores={scores_part.shape}, labels={len(label_space.labels)}"
            )

        train_rows, val_rows, split_info = make_split_from_plan(
            train_csv,
            soundscape_csv,
            split_cfg["data_root"],
            label_space,
            split_cfg,
        )
        if len(val_rows) != len(y_true_part):
            raise ValueError(
                "Saved predictions do not match reconstructed validation split length: "
                f"path={predictions_path}, predictions={len(y_true_part)}, "
                f"reconstructed_val_rows={len(val_rows)}, split={split_info.eval_split}, fold={split_info.fold}"
            )

        split_targets.append(y_true_part)
        split_scores.append(scores_part)
        train_rows_by_split.append(train_rows)
        soundscape_train_rows_by_split.append(train_rows[train_rows["source"] == "soundscape"])
        split_descriptions.append(
            f"{predictions_path.name}: eval_split={split_info.eval_split}, "
            f"fold={'n/a' if split_info.fold is None else split_info.fold}, rows={len(y_true_part)}"
        )

    y_true = np.concatenate(split_targets, axis=0)
    scores = np.concatenate(split_scores, axis=0)
    perch_mapping = pd.read_csv(cfg["perch_label_mapping_path"])

    per_class = build_per_class_breakdown(
        y_true,
        scores,
        label_space.labels,
        taxonomy,
        perch_mapping,
        train_rows_by_split,
        soundscape_train_rows_by_split,
    )
    groups = group_summary_from_per_class(per_class, label_space.labels, taxonomy, perch_mapping)

    per_class_path, group_path = default_output_paths(args.predictions, prediction_paths, args.output_dir)
    per_class_path.parent.mkdir(parents=True, exist_ok=True)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(per_class_path, index=False)
    groups.to_csv(group_path, index=False)

    print(f"Loaded {len(prediction_paths)} prediction file(s), total rows={len(y_true)}")
    for description in split_descriptions:
        print(f" - {description}")
    print(f"Wrote per-class AUC breakdown to {per_class_path}")
    print(f"Wrote group summary to {group_path}")


if __name__ == "__main__":
    main()
