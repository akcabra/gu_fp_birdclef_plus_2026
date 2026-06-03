from pathlib import Path
from pprint import pformat
from time import perf_counter
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import make_multicrop_tf_dataset, make_tf_dataset
from src.config import load_config
from src.data import (
    attach_targets,
    build_label_space,
    load_tables,
    make_split_from_plan,
    positive_class_weights,
    save_split,
    save_split_info,
)
from src.metrics import challenge_score_from_arrays, per_class_auc
from src.perch.blend import PerchScoreBlender
from src.perch.embedding_cache import load_embedding_cache, make_embedding_dataset, write_embedding_cache
from src.perch.model import build_embedding_model, build_model
from src.perch.train import format_duration, predict_dataset_scores, predict_multicrop_dataset, train_head_only
from src.utils import set_seed


def print_run_header(cfg: dict) -> None:
    print("Run configuration")
    print(pformat(cfg, sort_dicts=True))
    print()


def print_data_summary(train_rows, val_rows) -> None:
    print("Data")
    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(val_rows)}")
    print()


def make_loss_weights(cfg: dict, train_rows):
    if cfg["loss"] != "weighted_bce":
        return None
    weights = positive_class_weights(train_rows, cfg["weighted_bce_max_pos_weight"])
    print("Weighted BCE")
    print(f"positive weight min: {weights.min():.3f}")
    print(f"positive weight mean: {weights.mean():.3f}")
    print(f"positive weight max: {weights.max():.3f}")
    print()
    return weights


def make_perch_blender(cfg: dict, labels: list[str]) -> PerchScoreBlender | None:
    if not cfg["perch_blend_enabled"]:
        return None

    blender = PerchScoreBlender(
        labels,
        cfg["perch_label_mapping_path"],
        cfg["perch_blend_alpha"],
    )
    print("Perch score blend")
    print(f"alpha: {blender.alpha:.3f}")
    print(f"matched labels: {blender.matched_count}/{len(labels)}")
    print()
    return blender


def make_perch_mapper(cfg: dict, labels: list[str]) -> PerchScoreBlender:
    return PerchScoreBlender(labels, cfg["perch_label_mapping_path"], cfg["perch_blend_alpha"])


def validation_crop_offsets(cfg: dict) -> list[float]:
    num_crops = cfg["validation_num_crops"]
    stride = cfg["validation_crop_stride_seconds"]
    center = (num_crops - 1) / 2
    return [(i - center) * stride for i in range(num_crops)]


def expand_focal_rows(rows, crops_per_focal: int):
    if crops_per_focal < 1:
        raise ValueError("focal_train_crops_per_recording must be >= 1")
    rows = rows.reset_index(drop=True).copy()
    rows["source_row_index"] = range(len(rows))
    focal_rows = rows[rows["source"] == "focal"]
    other_rows = rows[rows["source"] != "focal"]
    parts = [focal_rows] * crops_per_focal + [other_rows]
    return pd.concat(parts, ignore_index=True)


def row_sampling_weights(cfg: dict, rows, labels: list[str], taxonomy, perch_mapping) -> pd.Series:
    weights = pd.Series(1.0, index=rows.index, dtype="float64")
    if cfg["sampling"] != "weighted":
        return weights

    train_counts = {label: 0 for label in labels}
    for row_labels in rows["labels"]:
        for label in row_labels:
            label = str(label)
            if label in train_counts:
                train_counts[label] += 1

    rare_labels = {label for label, count in train_counts.items() if 0 < count <= cfg["weighted_sampling_rare_threshold"]}
    matched_labels = set(perch_mapping.loc[perch_mapping["perch_index"].notna(), "primary_label"].astype(str))
    unmatched_labels = set(labels) - matched_labels
    taxonomy_by_label = taxonomy.set_index(taxonomy["primary_label"].astype(str))
    taxon_labels = set()
    for class_name in cfg["weighted_sampling_taxa"]:
        taxon_labels.update(taxonomy_by_label.loc[taxonomy_by_label["class_name"] == class_name].index.astype(str))

    for idx, row_labels in rows["labels"].items():
        row_label_set = {str(label) for label in row_labels}
        if row_label_set & rare_labels:
            weights.loc[idx] *= cfg["weighted_sampling_rare_multiplier"]
        if row_label_set & unmatched_labels:
            weights.loc[idx] *= cfg["weighted_sampling_unmatched_multiplier"]
        if row_label_set & taxon_labels:
            weights.loc[idx] *= cfg["weighted_sampling_taxa_multiplier"]

    return weights.clip(upper=cfg["weighted_sampling_max_weight"])


def print_sampling_summary(weights: pd.Series) -> None:
    print("Weighted sampling")
    print(f"sample weight min: {weights.min():.3f}")
    print(f"sample weight mean: {weights.mean():.3f}")
    print(f"sample weight max: {weights.max():.3f}")
    print()


def make_datasets(cfg: dict, train_rows, val_rows, labels: list[str], taxonomy, perch_mapping):
    offsets = validation_crop_offsets(cfg)
    expanded_train_rows = expand_focal_rows(train_rows, cfg["focal_train_crops_per_recording"])
    if not cfg["embedding_cache_enabled"]:
        sample_weights = row_sampling_weights(cfg, expanded_train_rows, labels, taxonomy, perch_mapping)
        if cfg["sampling"] == "weighted":
            print_sampling_summary(sample_weights)
        train_ds = make_tf_dataset(
            expanded_train_rows,
            batch_size=cfg["batch_size"],
            training=True,
            sample_weights=sample_weights.to_numpy(dtype="float64") if cfg["sampling"] == "weighted" else None,
            seed=cfg["seed"],
            augmentation=cfg["augmentation"],
        )
        if cfg["validation_num_crops"] == 1:
            val_ds = make_tf_dataset(val_rows, batch_size=cfg["batch_size"], training=False)
            val_row_indices = None
        else:
            val_ds, val_row_indices = make_multicrop_tf_dataset(val_rows, cfg["batch_size"], offsets)
            print("Validation crops")
            print(f"offsets_seconds: {offsets}")
            print()
        return train_ds, val_ds, val_row_indices, expanded_train_rows

    train_cache_path = Path(cfg["train_embedding_cache_path"])
    val_cache_path = Path(cfg["val_embedding_cache_path"])

    if cfg["write_embedding_cache"] or not train_cache_path.exists() or not val_cache_path.exists():
        raw_model = build_model(cfg)
        perch_mapper = make_perch_mapper(cfg, labels)

        train_raw_ds = make_tf_dataset(
            expanded_train_rows,
            batch_size=cfg["batch_size"],
            training=True,
            augmentation=cfg["augmentation"],
        )
        print(f"Writing train embedding cache to {train_cache_path}")
        write_embedding_cache(
            raw_model,
            train_raw_ds,
            train_cache_path,
            perch_mapper,
        )

        if cfg["validation_num_crops"] == 1:
            val_raw_ds = make_tf_dataset(val_rows, batch_size=cfg["batch_size"], training=False)
            val_row_indices = None
        else:
            val_raw_ds, val_row_indices = make_multicrop_tf_dataset(val_rows, cfg["batch_size"], offsets)
            print("Validation crops")
            print(f"offsets_seconds: {offsets}")
            print()
        print(f"Writing validation embedding cache to {val_cache_path}")
        write_embedding_cache(raw_model, val_raw_ds, val_cache_path, perch_mapper, val_row_indices)
        del raw_model

    train_cache = load_embedding_cache(train_cache_path)
    val_cache = load_embedding_cache(val_cache_path)
    if len(expanded_train_rows) != len(train_cache["targets"]):
        raise ValueError(
            "Embedding cache size does not match training rows. "
            "Regenerate the cache after changing focal_train_crops_per_recording or split settings."
        )
    sample_weights = row_sampling_weights(cfg, expanded_train_rows, labels, taxonomy, perch_mapping)
    if cfg["sampling"] == "weighted":
        print_sampling_summary(sample_weights)
    train_ds = make_embedding_dataset(
        train_cache,
        cfg["batch_size"],
        training=True,
        sample_weights=sample_weights.to_numpy(dtype="float64") if cfg["sampling"] == "weighted" else None,
        seed=cfg["seed"],
    )
    val_ds = make_embedding_dataset(val_cache, cfg["batch_size"], training=False)
    val_row_indices = val_cache["row_indices"] if "row_indices" in val_cache else None

    print("Embedding cache")
    print(f"train cache: {train_cache_path}")
    print(f"validation cache: {val_cache_path}")
    print(f"training examples: {len(train_cache['targets'])}")
    print(f"validation examples: {len(val_cache['targets'])}")
    print()
    return train_ds, val_ds, val_row_indices, expanded_train_rows


def best_epoch_from_history(history, phase: str) -> tuple[str, float] | None:
    if history is None:
        return None
    scores = history.history.get("val_challenge_score", [])
    if not scores:
        return None
    best_index = max(range(len(scores)), key=scores.__getitem__)
    return f"{phase} epoch {best_index + 1}", scores[best_index]


def best_epoch_from_histories(head_history) -> tuple[str, float]:
    return best_epoch_from_history(head_history, "head")


def labels_in_rows(rows) -> set[str]:
    labels = set()
    for row_labels in rows["labels"]:
        labels.update(str(label) for label in row_labels)
    return labels


def validation_group_summary(
    y_true,
    scores,
    labels: list[str],
    taxonomy,
    perch_mapping,
    train_rows,
    soundscape_train_rows,
) -> pd.DataFrame:
    per_class = per_class_auc(y_true, scores, labels)
    taxonomy_by_label = taxonomy.set_index(taxonomy["primary_label"].astype(str))
    matched_labels = set(perch_mapping.loc[perch_mapping["perch_index"].notna(), "primary_label"].astype(str))
    train_counts = {label: 0 for label in labels}
    for row_labels in train_rows["labels"]:
        for label in row_labels:
            label = str(label)
            if label in train_counts:
                train_counts[label] += 1
    soundscape_labels = labels_in_rows(soundscape_train_rows)

    groups = [
        ("all classes", set(labels)),
        ("matched Perch-label classes", matched_labels),
        ("unmatched Perch-label classes", set(labels) - matched_labels),
        ("rare classes train_count<=5", {label for label, count in train_counts.items() if 0 < count <= 5}),
        ("present in labeled train_soundscapes", soundscape_labels),
        ("absent from train_soundscapes", set(labels) - soundscape_labels),
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


def print_validation_group_summary(summary: pd.DataFrame) -> None:
    print()
    print("Validation group summary")
    out = summary.copy()
    out["mean_auc"] = out["mean_auc"].map(lambda value: "n/a" if pd.isna(value) else f"{value:.5f}")
    out["median_auc"] = out["median_auc"].map(lambda value: "n/a" if pd.isna(value) else f"{value:.5f}")
    print(out.to_string(index=False))


def print_experiment_summary(
    cfg: dict,
    head_history,
    final_score: float,
    training_seconds: float,
) -> None:
    best_result = best_epoch_from_histories(head_history)
    print()
    print("Experiment summary")
    print(f"experiment name: {cfg['experiment_name']}")
    print(f"seed: {cfg['seed']}")
    print(f"loss: {cfg['loss']}")
    print(f"sampling: {cfg['sampling']}")
    print(f"weighted_sampling_rare_threshold: {cfg['weighted_sampling_rare_threshold']}")
    print(f"weighted_sampling_rare_multiplier: {cfg['weighted_sampling_rare_multiplier']}")
    print(f"weighted_sampling_unmatched_multiplier: {cfg['weighted_sampling_unmatched_multiplier']}")
    print(f"weighted_sampling_taxa: {cfg['weighted_sampling_taxa']}")
    print(f"weighted_sampling_taxa_multiplier: {cfg['weighted_sampling_taxa_multiplier']}")
    print(f"weighted_sampling_max_weight: {cfg['weighted_sampling_max_weight']}")
    print(f"augmentation: {cfg['augmentation']}")
    print(f"focal_train_crops_per_recording: {cfg['focal_train_crops_per_recording']}")
    print(f"head_lr: {cfg['head_lr']}")
    print(f"dropout: {cfg['dropout']}")
    print(f"weighted_bce_max_pos_weight: {cfg['weighted_bce_max_pos_weight']}")
    print(f"focal_gamma: {cfg['focal_gamma']}")
    print(f"focal_alpha: {cfg['focal_alpha']}")
    print(f"validation_num_crops: {cfg['validation_num_crops']}")
    print(f"validation_crop_stride_seconds: {cfg['validation_crop_stride_seconds']}")
    print(f"validation_crop_aggregation: {cfg['validation_crop_aggregation']}")
    print(f"validation_top_k: {cfg['validation_top_k']}")
    print(f"perch_blend_enabled: {cfg['perch_blend_enabled']}")
    print(f"perch_blend_alpha: {cfg['perch_blend_alpha']}")
    print(f"embedding_cache_enabled: {cfg['embedding_cache_enabled']}")
    print(f"write_embedding_cache: {cfg['write_embedding_cache']}")
    print(f"train_embedding_cache_path: {cfg['train_embedding_cache_path']}")
    print(f"val_embedding_cache_path: {cfg['val_embedding_cache_path']}")
    print(f"train_enabled: {cfg['train_enabled']}")
    print(f"load_model_weights_path: {cfg['load_model_weights_path']}")
    print(f"save_model_weights_path: {cfg['save_model_weights_path']}")
    print(f"save_best_model: {cfg['save_best_model']}")
    print(f"best_model_weights_path: {cfg['best_model_weights_path']}")
    print(f"restore_best_model: {cfg['restore_best_model']}")
    if best_result is None:
        print("best epoch: n/a")
        print("best val_challenge_score: n/a")
    else:
        best_epoch, best_score = best_result
        print(f"best epoch: {best_epoch}")
        print(f"best val_challenge_score: {best_score:.5f}")
    print(f"final val_challenge_score: {final_score:.5f}")
    print(f"training time: {format_duration(training_seconds)}")
    print(f"notes: {cfg['notes']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="Path to a YAML config file")
    return parser.parse_args()


def main(config_path: str | Path = PROJECT_ROOT / "config.yaml"):
    run_start = perf_counter()
    cfg = load_config(config_path)
    print_run_header(cfg)
    set_seed(cfg["seed"])

    train_csv, soundscape_csv, taxonomy, sample_submission = load_tables(cfg["data_root"])
    label_space = build_label_space(sample_submission)

    train_rows, val_rows, split_info = make_split_from_plan(
        train_csv,
        soundscape_csv,
        cfg["data_root"],
        label_space,
        cfg,
    )
    train_rows = attach_targets(train_rows, label_space)
    val_rows = attach_targets(val_rows, label_space)
    print_data_summary(train_rows, val_rows)
    print(
        "Split: "
        f"eval_split={split_info.eval_split}, "
        f"fold={'n/a' if split_info.fold is None else split_info.fold}, "
        f"train_soundscape_files={split_info.train_soundscape_files}, "
        f"val_soundscape_files={split_info.val_soundscape_files}"
    )
    print(f"Split plan: {split_info.plan_path}")
    print()
    save_split(train_rows, val_rows, PROJECT_ROOT / "data" / "splits")
    save_split_info(split_info, PROJECT_ROOT / "data" / "splits")

    perch_mapping = pd.read_csv(cfg["perch_label_mapping_path"])
    train_ds, val_ds, val_row_indices, loss_weight_rows = make_datasets(
        cfg,
        train_rows,
        val_rows,
        label_space.labels,
        taxonomy,
        perch_mapping,
    )
    pos_weights = make_loss_weights(cfg, loss_weight_rows)
    perch_blender = make_perch_blender(cfg, label_space.labels)

    model = build_embedding_model(cfg) if cfg["embedding_cache_enabled"] else build_model(cfg)
    print("Model")
    model.summary(print_fn=print)
    print()

    if cfg["load_model_weights_path"]:
        model.load_weights(cfg["load_model_weights_path"], skip_mismatch=True)
        print(f"Loaded model weights from {cfg['load_model_weights_path']}")
        print()

    train_start = perf_counter()
    if cfg["train_enabled"]:
        head_history = train_head_only(
            model,
            train_ds,
            val_ds,
            label_space.labels,
            cfg,
            pos_weights,
            val_row_indices=val_row_indices,
            num_val_rows=len(val_rows),
            perch_blender=perch_blender,
        )
    else:
        print("Skipping training because train_enabled is false")
        head_history = None
    training_seconds = perf_counter() - train_start

    if cfg["save_model_weights_path"] and (cfg["train_enabled"] or cfg["load_model_weights_path"]):
        save_model_weights_path = Path(cfg["save_model_weights_path"])
        save_model_weights_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(save_model_weights_path)
        print(f"Saved model weights to {save_model_weights_path}")

    print(f"Total training time: {format_duration(training_seconds)}")

    eval_start = perf_counter()
    if val_row_indices is None:
        y_true, scores = predict_dataset_scores(model, val_ds, perch_blender)
    else:
        y_true, scores = predict_multicrop_dataset(
            model,
            val_ds,
            val_row_indices,
            len(val_rows),
            cfg["validation_crop_aggregation"],
            cfg["validation_top_k"],
            perch_blender,
        )
    score = challenge_score_from_arrays(y_true, scores, label_space.labels)
    print(f"Evaluation time: {format_duration(perf_counter() - eval_start)}")
    print(f"validation challenge score: {score:.5f}")
    if cfg.get("perch_val_predictions_path"):
        predictions_path = Path(cfg["perch_val_predictions_path"])
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.savez_compressed(
            predictions_path,
            targets=y_true,
            scores=scores,
            source_ids=val_rows["source_id"].astype(str).to_numpy(),
            split_eval=np.asarray(split_info.eval_split),
            split_fold=np.asarray(-1 if split_info.fold is None else split_info.fold),
            split_plan_path=np.asarray(split_info.plan_path),
        )
        print(f"Saved Perch validation predictions to {predictions_path}")
    soundscape_train_rows = train_rows[train_rows["source"] == "soundscape"]
    group_summary = validation_group_summary(
        y_true,
        scores,
        label_space.labels,
        taxonomy,
        perch_mapping,
        train_rows,
        soundscape_train_rows,
    )
    print_validation_group_summary(group_summary)
    print_experiment_summary(cfg, head_history, score, training_seconds)
    print(f"Total run time: {format_duration(perf_counter() - run_start)}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
