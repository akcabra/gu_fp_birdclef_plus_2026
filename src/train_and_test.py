from pathlib import Path
from pprint import pformat
from time import perf_counter
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import make_multicrop_tf_dataset, make_tf_dataset
from src.config import load_config
from src.data import attach_targets, build_label_space, load_tables, make_mixed_split, positive_class_weights, save_split
from src.embedding_cache import load_embedding_cache, make_embedding_dataset, write_embedding_cache
from src.metrics import challenge_score_from_arrays
from src.model import build_embedding_model, build_model
from src.perch_blend import PerchScoreBlender
from src.train import format_duration, predict_dataset_scores, predict_multicrop_dataset, train_head_only
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
    focal_rows = rows[rows["source"] == "focal"]
    other_rows = rows[rows["source"] != "focal"]
    parts = [focal_rows] * crops_per_focal + [other_rows]
    return pd.concat(parts, ignore_index=True)


def make_datasets(cfg: dict, train_rows, val_rows, labels: list[str]):
    offsets = validation_crop_offsets(cfg)
    if not cfg["embedding_cache_enabled"]:
        train_ds = make_tf_dataset(train_rows, batch_size=cfg["batch_size"], training=True)
        if cfg["validation_num_crops"] == 1:
            val_ds = make_tf_dataset(val_rows, batch_size=cfg["batch_size"], training=False)
            val_row_indices = None
        else:
            val_ds, val_row_indices = make_multicrop_tf_dataset(val_rows, cfg["batch_size"], offsets)
            print("Validation crops")
            print(f"offsets_seconds: {offsets}")
            print()
        return train_ds, val_ds, val_row_indices, train_rows

    cache_train_rows = expand_focal_rows(train_rows, cfg["embedding_cache_focal_crops_per_recording"])
    train_cache_path = Path(cfg["train_embedding_cache_path"])
    val_cache_path = Path(cfg["val_embedding_cache_path"])

    if cfg["write_embedding_cache"] or not train_cache_path.exists() or not val_cache_path.exists():
        raw_model = build_model(cfg)
        perch_mapper = make_perch_mapper(cfg, labels)

        train_raw_ds = make_tf_dataset(cache_train_rows, batch_size=cfg["batch_size"], training=True)
        print(f"Writing train embedding cache to {train_cache_path}")
        write_embedding_cache(raw_model, train_raw_ds, train_cache_path, perch_mapper)

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
    train_ds = make_embedding_dataset(train_cache, cfg["batch_size"], training=True)
    val_ds = make_embedding_dataset(val_cache, cfg["batch_size"], training=False)
    val_row_indices = val_cache["row_indices"] if "row_indices" in val_cache else None

    print("Embedding cache")
    print(f"train cache: {train_cache_path}")
    print(f"validation cache: {val_cache_path}")
    print(f"training examples: {len(train_cache['targets'])}")
    print(f"validation examples: {len(val_cache['targets'])}")
    print()
    return train_ds, val_ds, val_row_indices, cache_train_rows


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
    print(f"augmentation: {cfg['augmentation']}")
    print(f"head_lr: {cfg['head_lr']}")
    print(f"dropout: {cfg['dropout']}")
    print(f"weighted_bce_max_pos_weight: {cfg['weighted_bce_max_pos_weight']}")
    print(f"validation_num_crops: {cfg['validation_num_crops']}")
    print(f"validation_crop_stride_seconds: {cfg['validation_crop_stride_seconds']}")
    print(f"validation_crop_aggregation: {cfg['validation_crop_aggregation']}")
    print(f"validation_top_k: {cfg['validation_top_k']}")
    print(f"perch_blend_enabled: {cfg['perch_blend_enabled']}")
    print(f"perch_blend_alpha: {cfg['perch_blend_alpha']}")
    print(f"embedding_cache_enabled: {cfg['embedding_cache_enabled']}")
    print(f"write_embedding_cache: {cfg['write_embedding_cache']}")
    print(f"embedding_cache_focal_crops_per_recording: {cfg['embedding_cache_focal_crops_per_recording']}")
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


def main():
    run_start = perf_counter()
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    print_run_header(cfg)
    set_seed(cfg["seed"])

    train_csv, soundscape_csv, _, sample_submission = load_tables(cfg["data_root"])
    label_space = build_label_space(sample_submission)

    train_rows, val_rows = make_mixed_split(
        train_csv,
        soundscape_csv,
        cfg["data_root"],
        cfg["soundscape_val_fraction"],
        cfg["seed"],
    )
    train_rows = attach_targets(train_rows, label_space)
    val_rows = attach_targets(val_rows, label_space)
    print_data_summary(train_rows, val_rows)
    save_split(train_rows, val_rows, PROJECT_ROOT / "data" / "splits")

    train_ds, val_ds, val_row_indices, loss_weight_rows = make_datasets(cfg, train_rows, val_rows, label_space.labels)
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
    print_experiment_summary(cfg, head_history, score, training_seconds)
    print(f"Total run time: {format_duration(perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
