from __future__ import annotations

from pathlib import Path
from pprint import pformat
from time import perf_counter
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import attach_targets, build_label_space, load_tables, make_mixed_split, positive_class_weights, save_split
from src.metrics import challenge_score_from_arrays, per_class_auc, sigmoid
from src.passt.cache import load_passt_cache, write_passt_cache
from src.passt.progress import print_progress


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PaSSTHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float, head_type: str) -> None:
        super().__init__()
        if head_type == "mlp":
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 234),
            )
        elif head_type == "linear":
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, 234),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x):
        return self.net(x)


def print_run_header(cfg: dict) -> None:
    print_progress("Run configuration")
    print(pformat(cfg, sort_dicts=True), flush=True)
    print(flush=True)


def print_data_summary(train_rows, val_rows) -> None:
    print_progress("Data")
    print(f"Training examples: {len(train_rows)}", flush=True)
    print(f"Validation examples: {len(val_rows)}", flush=True)
    print(flush=True)


def validation_crop_offsets(cfg: dict) -> list[float]:
    num_crops = cfg["validation_num_crops"]
    stride = cfg["validation_crop_stride_seconds"]
    center = (num_crops - 1) / 2
    return [(i - center) * stride for i in range(num_crops)]


def expand_focal_rows(rows, crops_per_focal: int):
    rows = rows.reset_index(drop=True).copy()
    rows["source_row_index"] = range(len(rows))
    focal_rows = rows[rows["source"] == "focal"]
    other_rows = rows[rows["source"] != "focal"]
    parts = [focal_rows] * crops_per_focal + [other_rows]
    return pd.concat(parts, ignore_index=True)


def make_passt_val_rows(val_rows, offsets: list[float]):
    if len(offsets) == 1:
        rows = val_rows.reset_index(drop=True).copy()
        rows["row_index"] = range(len(rows))
        return rows

    expanded = []
    for row_idx, row in enumerate(val_rows.itertuples(index=False)):
        for offset in offsets:
            item = row._asdict()
            item["start_seconds"] = float(item["start_seconds"]) + offset
            item["row_index"] = row_idx
            expanded.append(item)
    return pd.DataFrame(expanded)


def labels_in_rows(rows) -> set[str]:
    labels = set()
    for row_labels in rows["labels"]:
        labels.update(str(label) for label in row_labels)
    return labels


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
    print_progress("Weighted sampling")
    print(f"sample weight min: {weights.min():.3f}", flush=True)
    print(f"sample weight mean: {weights.mean():.3f}", flush=True)
    print(f"sample weight max: {weights.max():.3f}", flush=True)
    print(flush=True)


def make_loss_weights(cfg: dict, train_rows):
    if cfg["loss"] != "weighted_bce":
        return None
    weights = positive_class_weights(train_rows, cfg["weighted_bce_max_pos_weight"])
    print_progress("Weighted BCE")
    print(f"positive weight min: {weights.min():.3f}", flush=True)
    print(f"positive weight mean: {weights.mean():.3f}", flush=True)
    print(f"positive weight max: {weights.max():.3f}", flush=True)
    print(flush=True)
    return weights


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
    print(flush=True)
    print_progress("Validation group summary")
    out = summary.copy()
    out["mean_auc"] = out["mean_auc"].map(lambda value: "n/a" if pd.isna(value) else f"{value:.5f}")
    out["median_auc"] = out["median_auc"].map(lambda value: "n/a" if pd.isna(value) else f"{value:.5f}")
    print(out.to_string(index=False), flush=True)


def aggregate_crop_scores(scores: np.ndarray, row_indices: np.ndarray, num_rows: int, aggregation: str, top_k: int) -> np.ndarray:
    output = np.zeros((num_rows, scores.shape[1]), dtype=np.float32)
    for row_idx in range(num_rows):
        row_scores = scores[row_indices == row_idx]
        if aggregation == "mean":
            output[row_idx] = row_scores.mean(axis=0)
        elif aggregation == "max":
            output[row_idx] = row_scores.max(axis=0)
        elif aggregation == "top_k_mean":
            k = min(top_k, len(row_scores))
            output[row_idx] = np.sort(row_scores, axis=0)[-k:].mean(axis=0)
        else:
            raise ValueError(f"Unknown validation_crop_aggregation: {aggregation}")
    return output


def build_group_candidates(source_row_indices: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    group_ids = np.unique(source_row_indices)
    candidates = [np.flatnonzero(source_row_indices == group_id) for group_id in group_ids]
    return group_ids, candidates


def iter_train_batches(
    embeddings: np.ndarray,
    targets: np.ndarray,
    source_row_indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    sample_weights: np.ndarray | None,
):
    group_ids, candidates = build_group_candidates(source_row_indices)
    if sample_weights is None:
        group_order = rng.permutation(len(group_ids))
    else:
        group_weights = sample_weights[group_ids]
        probabilities = group_weights / group_weights.sum()
        group_order = rng.choice(len(group_ids), size=len(group_ids), replace=True, p=probabilities)

    for start in range(0, len(group_order), batch_size):
        batch_groups = group_order[start : start + batch_size]
        batch_indices = [rng.choice(candidates[group_idx]) for group_idx in batch_groups]
        yield embeddings[batch_indices], targets[batch_indices]


def torch_loss(logits, targets, cfg: dict, pos_weights):
    if cfg["loss"] == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if cfg["loss"] == "weighted_bce":
        if pos_weights is None:
            raise ValueError("weighted_bce requires positive class weights")
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weights)
    if cfg["loss"] == "focal":
        probs = torch.sigmoid(logits)
        cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        loss = torch.pow(1.0 - p_t, cfg["focal_gamma"]) * cross_entropy
        alpha = cfg["focal_alpha"]
        if alpha is not None:
            alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
            loss = alpha_t * loss
        return loss.mean()
    raise ValueError(f"Unknown loss: {cfg['loss']}")


def predict_arrays(model, embeddings: np.ndarray, targets: np.ndarray, batch_size: int, device: torch.device):
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            x = torch.from_numpy(embeddings[start : start + batch_size]).to(device)
            logits.append(model(x).cpu().numpy())
    scores = sigmoid(np.concatenate(logits, axis=0))
    return targets.astype(np.float32), scores.astype(np.float32)


def save_checkpoint(path: str | Path, model, cfg: dict, embedding_dim: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "head_type": cfg["head_type"],
            "hidden_dim": cfg["hidden_dim"],
            "dropout": cfg["dropout"],
        },
        path,
    )


def load_checkpoint(path: str | Path, model, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def make_passt_caches_if_needed(cfg: dict, train_rows, val_rows):
    offsets = validation_crop_offsets(cfg)
    input_samples = int(32000 * cfg["passt_input_seconds"])
    train_cache_path = Path(cfg["passt_train_cache_path"])
    val_cache_path = Path(cfg["passt_val_cache_path"])
    if not cfg["write_passt_cache"] and train_cache_path.exists() and val_cache_path.exists():
        return

    cache_train_rows = expand_focal_rows(train_rows, cfg["passt_cache_focal_crops_per_recording"])
    val_cache_rows = make_passt_val_rows(val_rows, offsets)
    if len(offsets) > 1:
        print_progress("Validation crops")
        print(f"offsets_seconds: {offsets}", flush=True)
        print(flush=True)

    print_progress(f"Writing PaSST train embedding cache to {train_cache_path}")
    write_passt_cache(
        cache_train_rows,
        train_cache_path,
        batch_size=cfg["passt_batch_size"],
        device=cfg["passt_device"],
        arch=cfg["passt_arch"],
        include_logits=cfg["passt_cache_include_logits"],
        input_samples=input_samples,
    )
    print_progress(f"Writing PaSST validation embedding cache to {val_cache_path}")
    write_passt_cache(
        val_cache_rows,
        val_cache_path,
        batch_size=cfg["passt_batch_size"],
        device=cfg["passt_device"],
        arch=cfg["passt_arch"],
        include_logits=cfg["passt_cache_include_logits"],
        input_samples=input_samples,
    )


def main():
    run_start = perf_counter()
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    print_run_header(cfg)
    set_seed(cfg["seed"])

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
    val_rows = attach_targets(val_rows, label_space)
    print_data_summary(train_rows, val_rows)
    save_split(train_rows, val_rows, PROJECT_ROOT / "data" / "splits")

    perch_mapping = pd.read_csv(cfg["perch_label_mapping_path"])
    make_passt_caches_if_needed(cfg, train_rows, val_rows)

    train_cache = load_passt_cache(cfg["passt_train_cache_path"])
    val_cache = load_passt_cache(cfg["passt_val_cache_path"])
    if "source_row_indices" not in train_cache:
        raise ValueError("PaSST training cache does not include source_row_indices. Regenerate the cache.")

    train_embeddings = train_cache["embeddings"].astype(np.float32)
    train_targets = train_cache["targets"].astype(np.float32)
    source_row_indices = train_cache["source_row_indices"].astype(np.int32)
    val_embeddings = val_cache["embeddings"].astype(np.float32)
    val_targets = val_cache["targets"].astype(np.float32)
    val_row_indices = val_cache["row_indices"] if "row_indices" in val_cache else None
    embedding_dim = train_embeddings.shape[1]

    print_progress("PaSST embedding cache")
    print(f"train cache: {cfg['passt_train_cache_path']}", flush=True)
    print(f"validation cache: {cfg['passt_val_cache_path']}", flush=True)
    print(f"training examples: {len(train_targets)}", flush=True)
    print(f"validation examples: {len(val_targets)}", flush=True)
    print(f"embedding dim: {embedding_dim}", flush=True)
    print(flush=True)

    sample_weights = row_sampling_weights(cfg, train_rows.reset_index(drop=True), label_space.labels, taxonomy, perch_mapping)
    if cfg["sampling"] == "weighted":
        print_sampling_summary(sample_weights)
        sample_weight_array = sample_weights.to_numpy(dtype=np.float64)
    else:
        sample_weight_array = None

    pos_weights_np = make_loss_weights(cfg, expand_focal_rows(train_rows, cfg["passt_cache_focal_crops_per_recording"]))
    device = torch.device(cfg["passt_device"] if cfg["passt_device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print_progress(f"PyTorch device: {device}")
    print(flush=True)

    model = PaSSTHead(embedding_dim, cfg["hidden_dim"], cfg["dropout"], cfg["head_type"]).to(device)
    print_progress("Model")
    print(model, flush=True)
    print(flush=True)

    if cfg["passt_load_model_weights_path"]:
        load_checkpoint(cfg["passt_load_model_weights_path"], model, device)
        print_progress(f"Loaded PaSST head weights from {cfg['passt_load_model_weights_path']}")
        print(flush=True)

    pos_weights = None if pos_weights_np is None else torch.from_numpy(pos_weights_np).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["head_lr"])
    rng = np.random.default_rng(cfg["seed"])
    best_score = -np.inf
    best_epoch = None

    train_start = perf_counter()
    if cfg["train_enabled"]:
        print_progress("Starting PaSST head training")
        for epoch in range(cfg["epochs_head"]):
            epoch_start = perf_counter()
            model.train()
            losses = []
            for x_np, y_np in iter_train_batches(
                train_embeddings,
                train_targets,
                source_row_indices,
                cfg["batch_size"],
                rng,
                sample_weight_array,
            ):
                x = torch.from_numpy(x_np).to(device)
                y = torch.from_numpy(y_np).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = torch_loss(logits, y, cfg, pos_weights)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            crop_true, crop_scores = predict_arrays(model, val_embeddings, val_targets, cfg["batch_size"], device)
            if val_row_indices is None:
                y_true, scores = crop_true, crop_scores
            else:
                scores = aggregate_crop_scores(
                    crop_scores,
                    val_row_indices,
                    len(val_rows),
                    cfg["validation_crop_aggregation"],
                    cfg["validation_top_k"],
                )
                y_true = np.zeros((len(val_rows), crop_true.shape[1]), dtype=np.float32)
                for row_idx in range(len(val_rows)):
                    y_true[row_idx] = crop_true[np.flatnonzero(val_row_indices == row_idx)[0]]
            score = challenge_score_from_arrays(y_true, scores, label_space.labels)
            print_progress(f"Epoch {epoch + 1}/{cfg['epochs_head']}")
            print(f" - epoch_time: {format_duration(perf_counter() - epoch_start)}", flush=True)
            print(f" - loss: {np.mean(losses):.6f}", flush=True)
            print(f" - val_challenge_score: {score:.5f}", flush=True)

            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                if cfg["passt_save_best_model"]:
                    save_checkpoint(cfg["passt_best_model_weights_path"], model, cfg, embedding_dim)
        if cfg["restore_best_model"] and cfg["passt_save_best_model"] and cfg["passt_best_model_weights_path"]:
            load_checkpoint(cfg["passt_best_model_weights_path"], model, device)
            print_progress(f"Loaded best PaSST head weights from {cfg['passt_best_model_weights_path']}")
    else:
        print_progress("Skipping training because train_enabled is false")
    training_seconds = perf_counter() - train_start

    if cfg["passt_save_model_weights_path"] and (cfg["train_enabled"] or cfg["passt_load_model_weights_path"]):
        save_checkpoint(cfg["passt_save_model_weights_path"], model, cfg, embedding_dim)
        print_progress(f"Saved PaSST head weights to {cfg['passt_save_model_weights_path']}")

    print_progress(f"Total training time: {format_duration(training_seconds)}")

    eval_start = perf_counter()
    crop_true, crop_scores = predict_arrays(model, val_embeddings, val_targets, cfg["batch_size"], device)
    if val_row_indices is None:
        y_true, scores = crop_true, crop_scores
    else:
        scores = aggregate_crop_scores(
            crop_scores,
            val_row_indices,
            len(val_rows),
            cfg["validation_crop_aggregation"],
            cfg["validation_top_k"],
        )
        y_true = np.zeros((len(val_rows), crop_true.shape[1]), dtype=np.float32)
        for row_idx in range(len(val_rows)):
            y_true[row_idx] = crop_true[np.flatnonzero(val_row_indices == row_idx)[0]]
    score = challenge_score_from_arrays(y_true, scores, label_space.labels)
    print_progress(f"Evaluation time: {format_duration(perf_counter() - eval_start)}")
    print(f"validation challenge score: {score:.5f}", flush=True)

    if cfg["passt_val_predictions_path"]:
        predictions_path = Path(cfg["passt_val_predictions_path"])
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(predictions_path, targets=y_true, scores=scores)
        print_progress(f"Saved PaSST validation predictions to {predictions_path}")

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

    print(flush=True)
    print_progress("Experiment summary")
    print(f"experiment name: {cfg['experiment_name']}", flush=True)
    print(f"seed: {cfg['seed']}", flush=True)
    print(f"loss: {cfg['loss']}", flush=True)
    print(f"sampling: {cfg['sampling']}", flush=True)
    print(f"head_lr: {cfg['head_lr']}", flush=True)
    print(f"dropout: {cfg['dropout']}", flush=True)
    print(f"focal_gamma: {cfg['focal_gamma']}", flush=True)
    print(f"focal_alpha: {cfg['focal_alpha']}", flush=True)
    print(f"passt_input_seconds: {cfg['passt_input_seconds']}", flush=True)
    print(f"passt_cache_focal_crops_per_recording: {cfg['passt_cache_focal_crops_per_recording']}", flush=True)
    print(f"passt_train_cache_path: {cfg['passt_train_cache_path']}", flush=True)
    print(f"passt_val_cache_path: {cfg['passt_val_cache_path']}", flush=True)
    print(f"train_enabled: {cfg['train_enabled']}", flush=True)
    print(f"passt_load_model_weights_path: {cfg['passt_load_model_weights_path']}", flush=True)
    print(f"passt_save_model_weights_path: {cfg['passt_save_model_weights_path']}", flush=True)
    print(f"passt_save_best_model: {cfg['passt_save_best_model']}", flush=True)
    print(f"passt_best_model_weights_path: {cfg['passt_best_model_weights_path']}", flush=True)
    if best_epoch is None:
        print("best epoch: n/a", flush=True)
        print("best val_challenge_score: n/a", flush=True)
    else:
        print(f"best epoch: head epoch {best_epoch}", flush=True)
        print(f"best val_challenge_score: {best_score:.5f}", flush=True)
    print(f"final val_challenge_score: {score:.5f}", flush=True)
    print(f"training time: {format_duration(training_seconds)}", flush=True)
    print(f"notes: {cfg['notes']}", flush=True)
    print_progress(f"Total run time: {format_duration(perf_counter() - run_start)}")


if __name__ == "__main__":
    main()
