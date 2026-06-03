from __future__ import annotations

from pathlib import Path
from pprint import pformat
from time import perf_counter
import contextlib
import io
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import load_clip_np
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


class PaSSTWaveformDataset(Dataset):
    def __init__(self, rows, training: bool, augmentation: str = "random_crop", row_indices: np.ndarray | None = None) -> None:
        self.paths = rows["audio_path"].astype(str).to_numpy()
        self.sources = rows["source"].astype(str).to_numpy()
        self.source_ids = np.asarray([0 if source == "focal" else 1 for source in self.sources], dtype=np.int64)
        self.starts = rows["start_seconds"].fillna(-1).astype(np.float32).to_numpy()
        self.targets = np.stack(rows["target"].to_numpy()).astype(np.float32)
        self.training = training
        self.augmentation = augmentation
        self.row_indices = row_indices

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        waveform = load_clip_np(
            self.paths[index],
            self.sources[index],
            float(self.starts[index]),
            self.training,
            self.augmentation,
        )
        target = self.targets[index]
        source_id = self.source_ids[index]
        if self.row_indices is None:
            return waveform, target, source_id
        return waveform, target, source_id, np.int32(self.row_indices[index])


class OnlinePaSSTClassifier(nn.Module):
    def __init__(self, cfg: dict, device: torch.device, num_classes: int = 234) -> None:
        super().__init__()
        from hear21passt import base as passt_base

        with contextlib.redirect_stdout(io.StringIO()):
            if cfg["passt_arch"]:
                self.passt = passt_base.get_basic_model(arch=cfg["passt_arch"])
            else:
                self.passt = passt_base.load_model()
        self.passt.to(device)
        self.feature_mode = cfg["passt_feature_mode"]
        self.embedding_dim = self._embedding_dim()
        self.head = self._make_head(self.embedding_dim, cfg["hidden_dim"], cfg["dropout"], cfg["head_type"], num_classes)
        self._configure_trainable_backbone(cfg)

    def _embedding_dim(self) -> int:
        if self.feature_mode == "all":
            return 1295
        if self.feature_mode == "features":
            return 768
        if self.feature_mode == "logits":
            return 527
        raise ValueError(f"Unknown passt_feature_mode: {self.feature_mode}")

    @staticmethod
    def _make_head(embedding_dim: int, hidden_dim: int, dropout: float, head_type: str, num_classes: int) -> nn.Module:
        if head_type == "mlp":
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        if head_type == "linear":
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, num_classes),
            )
        raise ValueError(f"Unknown head_type: {head_type}")

    def _configure_trainable_backbone(self, cfg: dict) -> None:
        for parameter in self.passt.parameters():
            parameter.requires_grad = False

        if not cfg["passt_train_backbone"]:
            return

        last_n_blocks = int(cfg["passt_unfreeze_last_n_blocks"])
        if last_n_blocks > 0:
            blocks = list(self.passt.net.blocks)
            for block in blocks[-last_n_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            for parameter in self.passt.net.norm.parameters():
                parameter.requires_grad = True
        else:
            for parameter in self.passt.parameters():
                parameter.requires_grad = True

    def features_from_specs(self, specs: torch.Tensor) -> torch.Tensor:
        self.passt.net.patch_embed.img_size = (int(specs.shape[1]), int(specs.shape[2]))
        logits, features = self.passt.net(specs.unsqueeze(1))
        if self.feature_mode == "all":
            return torch.cat([logits, features], dim=1)
        if self.feature_mode == "features":
            return features
        if self.feature_mode == "logits":
            return logits
        raise ValueError(f"Unknown passt_feature_mode: {self.feature_mode}")

    def forward_from_specs(self, specs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features_from_specs(specs))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        specs = self.passt.mel(waveforms)
        return self.forward_from_specs(specs)


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


def make_passt_val_rows_with_indices(val_rows, offsets: list[float]) -> tuple[pd.DataFrame, np.ndarray | None]:
    if len(offsets) == 1:
        return val_rows.reset_index(drop=True).copy(), None

    rows = make_passt_val_rows(val_rows, offsets)
    row_indices = rows["row_index"].to_numpy(dtype=np.int32)
    rows = rows.drop(columns=["row_index"])
    return rows, row_indices


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


def mixup_partner_indices(source_ids: torch.Tensor, mode: str) -> torch.Tensor | None:
    indices = torch.arange(source_ids.shape[0], device=source_ids.device)
    if mode == "batch":
        return indices[torch.randperm(len(indices), device=source_ids.device)]
    if mode == "same_source":
        partner_indices = indices.clone()
        for source_id in torch.unique(source_ids):
            group = indices[source_ids == source_id]
            if len(group) > 1:
                partner_indices[group] = group[torch.randperm(len(group), device=source_ids.device)]
        return partner_indices
    if mode == "focal_only":
        group = indices[source_ids == 0]
    elif mode == "soundscape_only":
        group = indices[source_ids == 1]
    else:
        raise ValueError(f"Unknown passt_mixup_mode: {mode}")

    if len(group) < 2:
        return None
    partner_indices = indices.clone()
    partner_indices[group] = group[torch.randperm(len(group), device=source_ids.device)]
    return partner_indices


def apply_mixup(
    specs: torch.Tensor,
    targets: torch.Tensor,
    source_ids: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = cfg["passt_mixup_alpha"]
    if alpha <= 0:
        return specs, targets
    if np.random.random() > cfg["passt_mixup_probability"]:
        return specs, targets

    indices = mixup_partner_indices(source_ids, cfg["passt_mixup_mode"])
    if indices is None:
        return specs, targets

    lam = torch.distributions.Beta(alpha, alpha).sample((specs.shape[0],)).to(specs.device)
    if cfg["passt_mixup_mode"] == "focal_only":
        lam = torch.where(source_ids == 0, lam, torch.ones_like(lam))
    elif cfg["passt_mixup_mode"] == "soundscape_only":
        lam = torch.where(source_ids == 1, lam, torch.ones_like(lam))

    spec_lam = lam.view(-1, 1, 1)
    target_lam = lam.view(-1, 1)
    mixed_specs = spec_lam * specs + (1.0 - spec_lam) * specs[indices]
    if cfg["passt_mixup_target_mode"] == "linear":
        mixed_targets = target_lam * targets + (1.0 - target_lam) * targets[indices]
    elif cfg["passt_mixup_target_mode"] == "max":
        mixed_targets = torch.maximum(targets, targets[indices])
    else:
        raise ValueError(f"Unknown passt_mixup_target_mode: {cfg['passt_mixup_target_mode']}")
    return mixed_specs, mixed_targets


def apply_specaugment(specs: torch.Tensor, time_mask: int, freq_mask: int) -> torch.Tensor:
    if time_mask <= 0 and freq_mask <= 0:
        return specs
    specs = specs.clone()
    batch_size, n_mels, n_frames = specs.shape
    if freq_mask > 0 and n_mels > 1:
        width = min(freq_mask, n_mels)
        starts = torch.randint(0, n_mels - width + 1, (batch_size,), device=specs.device)
        for batch_idx, start in enumerate(starts):
            specs[batch_idx, start : start + width, :] = 0
    if time_mask > 0 and n_frames > 1:
        width = min(time_mask, n_frames)
        starts = torch.randint(0, n_frames - width + 1, (batch_size,), device=specs.device)
        for batch_idx, start in enumerate(starts):
            specs[batch_idx, :, start : start + width] = 0
    return specs


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


def predict_online(model, loader: DataLoader, device: torch.device):
    model.eval()
    targets = []
    logits = []
    row_indices = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x, y, _, rows = batch
                row_indices.append(rows.numpy())
            else:
                x, y, _ = batch
            x = x.to(device)
            logits.append(model(x).cpu().numpy())
            targets.append(y.numpy())
    scores = sigmoid(np.concatenate(logits, axis=0))
    y_true = np.concatenate(targets, axis=0).astype(np.float32)
    if row_indices:
        return y_true, scores.astype(np.float32), np.concatenate(row_indices, axis=0)
    return y_true, scores.astype(np.float32), None


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
    cache_train_rows = cache_train_rows.copy()
    cache_train_rows["augmentation"] = cfg["augmentation"]
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


def make_online_optimizer(model: OnlinePaSSTClassifier, cfg: dict):
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    backbone_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("head.") and param.requires_grad
    ]
    groups = [{"params": head_params, "lr": cfg["head_lr"]}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg["passt_backbone_lr"]})
    return torch.optim.Adam(groups)


def run_online_training(
    cfg: dict,
    train_rows,
    val_rows,
    label_space,
    taxonomy,
    perch_mapping,
    sample_weights: pd.Series,
    pos_weights_np,
    device: torch.device,
    run_start: float,
) -> None:
    offsets = validation_crop_offsets(cfg)
    val_eval_rows, val_row_indices = make_passt_val_rows_with_indices(val_rows, offsets)
    if val_row_indices is not None:
        print_progress("Validation crops")
        print(f"offsets_seconds: {offsets}", flush=True)
        print(flush=True)

    train_dataset = PaSSTWaveformDataset(train_rows.reset_index(drop=True), training=True, augmentation=cfg["augmentation"])
    val_dataset = PaSSTWaveformDataset(val_eval_rows.reset_index(drop=True), training=False, row_indices=val_row_indices)
    sampler = None
    shuffle = True
    if cfg["sampling"] == "weighted":
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights.to_numpy(dtype=np.float64)),
            num_samples=len(train_dataset),
            replacement=True,
        )
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg["passt_num_workers"],
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["passt_batch_size"],
        shuffle=False,
        num_workers=cfg["passt_num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = OnlinePaSSTClassifier(cfg, device).to(device)
    print_progress("Model")
    print(model, flush=True)
    trainable_backbone = sum(
        int(param.numel()) for name, param in model.named_parameters() if not name.startswith("head.") and param.requires_grad
    )
    trainable_head = sum(int(param.numel()) for param in model.head.parameters() if param.requires_grad)
    print(f"trainable head params: {trainable_head}", flush=True)
    print(f"trainable PaSST backbone params: {trainable_backbone}", flush=True)
    print(flush=True)

    if cfg["passt_load_model_weights_path"]:
        load_checkpoint(cfg["passt_load_model_weights_path"], model, device)
        print_progress(f"Loaded PaSST model weights from {cfg['passt_load_model_weights_path']}")
        print(flush=True)

    pos_weights = None if pos_weights_np is None else torch.from_numpy(pos_weights_np).to(device)
    optimizer = make_online_optimizer(model, cfg)
    best_score = -np.inf
    best_epoch = None

    train_start = perf_counter()
    if cfg["train_enabled"]:
        print_progress("Starting online PaSST training")
        for epoch in range(cfg["epochs_head"]):
            epoch_start = perf_counter()
            model.train()
            model.passt.mel.eval()
            losses = []
            for x, y, source_ids in train_loader:
                x = x.to(device)
                y = y.to(device)
                source_ids = source_ids.to(device)
                optimizer.zero_grad(set_to_none=True)
                specs = model.passt.mel(x)
                if cfg["passt_mixup_enabled"]:
                    specs, y = apply_mixup(specs, y, source_ids, cfg)
                if cfg["passt_specaugment_enabled"]:
                    specs = apply_specaugment(
                        specs,
                        cfg["passt_specaugment_time_mask"],
                        cfg["passt_specaugment_freq_mask"],
                    )
                logits = model.forward_from_specs(specs)
                loss = torch_loss(logits, y, cfg, pos_weights)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            crop_true, crop_scores, eval_row_indices = predict_online(model, val_loader, device)
            if eval_row_indices is None:
                y_true, scores = crop_true, crop_scores
            else:
                scores = aggregate_crop_scores(
                    crop_scores,
                    eval_row_indices,
                    len(val_rows),
                    cfg["validation_crop_aggregation"],
                    cfg["validation_top_k"],
                )
                y_true = np.zeros((len(val_rows), crop_true.shape[1]), dtype=np.float32)
                for row_idx in range(len(val_rows)):
                    y_true[row_idx] = crop_true[np.flatnonzero(eval_row_indices == row_idx)[0]]
            score = challenge_score_from_arrays(y_true, scores, label_space.labels)
            print_progress(f"Epoch {epoch + 1}/{cfg['epochs_head']}")
            print(f" - epoch_time: {format_duration(perf_counter() - epoch_start)}", flush=True)
            print(f" - loss: {np.mean(losses):.6f}", flush=True)
            print(f" - val_challenge_score: {score:.5f}", flush=True)
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                if cfg["passt_save_best_model"]:
                    save_checkpoint(cfg["passt_best_model_weights_path"], model, cfg, model.embedding_dim)
        if cfg["restore_best_model"] and cfg["passt_save_best_model"] and cfg["passt_best_model_weights_path"]:
            load_checkpoint(cfg["passt_best_model_weights_path"], model, device)
            print_progress(f"Loaded best PaSST model weights from {cfg['passt_best_model_weights_path']}")
    else:
        print_progress("Skipping training because train_enabled is false")
    training_seconds = perf_counter() - train_start

    if cfg["passt_save_model_weights_path"] and (cfg["train_enabled"] or cfg["passt_load_model_weights_path"]):
        save_checkpoint(cfg["passt_save_model_weights_path"], model, cfg, model.embedding_dim)
        print_progress(f"Saved PaSST model weights to {cfg['passt_save_model_weights_path']}")

    print_progress(f"Total training time: {format_duration(training_seconds)}")
    eval_start = perf_counter()
    crop_true, crop_scores, eval_row_indices = predict_online(model, val_loader, device)
    if eval_row_indices is None:
        y_true, scores = crop_true, crop_scores
    else:
        scores = aggregate_crop_scores(
            crop_scores,
            eval_row_indices,
            len(val_rows),
            cfg["validation_crop_aggregation"],
            cfg["validation_top_k"],
        )
        y_true = np.zeros((len(val_rows), crop_true.shape[1]), dtype=np.float32)
        for row_idx in range(len(val_rows)):
            y_true[row_idx] = crop_true[np.flatnonzero(eval_row_indices == row_idx)[0]]
    score = challenge_score_from_arrays(y_true, scores, label_space.labels)
    print_progress(f"Evaluation time: {format_duration(perf_counter() - eval_start)}")
    print(f"validation challenge score: {score:.5f}", flush=True)

    if cfg["passt_val_predictions_path"]:
        predictions_path = Path(cfg["passt_val_predictions_path"])
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            predictions_path,
            targets=y_true,
            scores=scores,
            source_ids=val_rows["source_id"].astype(str).to_numpy(),
            split_eval=np.asarray(cfg["soundscape_eval_split"]),
            split_fold=np.asarray(int(cfg["soundscape_fold"]) if cfg["soundscape_eval_split"] == "cv" else -1),
            split_plan_path=np.asarray(cfg["soundscape_split_plan_path"]),
        )
        print_progress(f"Saved PaSST validation predictions to {predictions_path}")

    soundscape_train_rows = train_rows[train_rows["source"] == "soundscape"]
    print_validation_group_summary(
        validation_group_summary(y_true, scores, label_space.labels, taxonomy, perch_mapping, train_rows, soundscape_train_rows)
    )

    print(flush=True)
    print_progress("Experiment summary")
    print(f"experiment name: {cfg['experiment_name']}", flush=True)
    print(f"seed: {cfg['seed']}", flush=True)
    print(f"loss: {cfg['loss']}", flush=True)
    print(f"sampling: {cfg['sampling']}", flush=True)
    print(f"passt_training_mode: {cfg['passt_training_mode']}", flush=True)
    print(f"passt_train_backbone: {cfg['passt_train_backbone']}", flush=True)
    print(f"passt_unfreeze_last_n_blocks: {cfg['passt_unfreeze_last_n_blocks']}", flush=True)
    print(f"passt_mixup_enabled: {cfg['passt_mixup_enabled']}", flush=True)
    print(f"passt_mixup_mode: {cfg['passt_mixup_mode']}", flush=True)
    print(f"passt_mixup_target_mode: {cfg['passt_mixup_target_mode']}", flush=True)
    print(f"passt_mixup_probability: {cfg['passt_mixup_probability']}", flush=True)
    print(f"passt_specaugment_enabled: {cfg['passt_specaugment_enabled']}", flush=True)
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


def main():
    run_start = perf_counter()
    cfg = load_config(PROJECT_ROOT / "config.yaml")
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
        f"val_soundscape_files={split_info.val_soundscape_files}",
        flush=True,
    )
    print(f"Split plan: {split_info.plan_path}", flush=True)
    print(flush=True)
    save_split(train_rows, val_rows, PROJECT_ROOT / "data" / "splits")
    save_split_info(split_info, PROJECT_ROOT / "data" / "splits")

    perch_mapping = pd.read_csv(cfg["perch_label_mapping_path"])
    sample_weights = row_sampling_weights(cfg, train_rows.reset_index(drop=True), label_space.labels, taxonomy, perch_mapping)
    if cfg["sampling"] == "weighted":
        print_sampling_summary(sample_weights)
        sample_weight_array = sample_weights.to_numpy(dtype=np.float64)
    else:
        sample_weight_array = None
    pos_weight_rows = (
        expand_focal_rows(train_rows, cfg["passt_cache_focal_crops_per_recording"])
        if cfg["passt_training_mode"] == "cache"
        else train_rows
    )
    pos_weights_np = make_loss_weights(cfg, pos_weight_rows)
    device = torch.device(cfg["passt_device"] if cfg["passt_device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print_progress(f"PyTorch device: {device}")
    print(flush=True)

    if cfg["passt_training_mode"] == "online":
        run_online_training(
            cfg,
            train_rows,
            val_rows,
            label_space,
            taxonomy,
            perch_mapping,
            sample_weights,
            pos_weights_np,
            device,
            run_start,
        )
        return
    if cfg["passt_training_mode"] != "cache":
        raise ValueError(f"Unknown passt_training_mode: {cfg['passt_training_mode']}")

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
        np.savez_compressed(
            predictions_path,
            targets=y_true,
            scores=scores,
            source_ids=val_rows["source_id"].astype(str).to_numpy(),
            split_eval=np.asarray(split_info.eval_split),
            split_fold=np.asarray(-1 if split_info.fold is None else split_info.fold),
            split_plan_path=np.asarray(split_info.plan_path),
        )
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
