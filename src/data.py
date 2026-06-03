from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpace:
    labels: list[str]
    label_to_idx: dict[str, int]
    idx_to_label: dict[int, str]


@dataclass(frozen=True)
class SplitInfo:
    eval_split: str
    fold: int | None
    plan_path: str
    train_soundscape_files: int
    val_soundscape_files: int


def load_tables(data_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_root = Path(data_root)
    train = pd.read_csv(data_root / "train.csv")
    soundscape = pd.read_csv(data_root / "train_soundscapes_labels.csv")
    taxonomy = pd.read_csv(data_root / "taxonomy.csv")
    sample_submission = pd.read_csv(data_root / "sample_submission.csv")
    return train, soundscape, taxonomy, sample_submission


def build_label_space(sample_submission: pd.DataFrame) -> LabelSpace:
    labels = [str(c) for c in sample_submission.columns[1:]]
    return LabelSpace(
        labels=labels,
        label_to_idx={label: i for i, label in enumerate(labels)},
        idx_to_label={i: label for i, label in enumerate(labels)},
    )


def parse_time_to_seconds(value: str) -> float:
    parts = str(value).split(":")
    if len(parts) != 3:
        raise ValueError(f"Expected HH:MM:SS time, got {value!r}")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_soundscape_labels(value: str) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [label.strip() for label in str(value).split(";") if label.strip()]


def multi_hot(labels: list[str], label_space: LabelSpace) -> np.ndarray:
    target = np.zeros(len(label_space.labels), dtype=np.float32)
    for label in labels:
        idx = label_space.label_to_idx.get(str(label))
        if idx is not None:
            target[idx] = 1.0
    return target


def make_focal_rows(train: pd.DataFrame, data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    rows = pd.DataFrame(
        {
            "audio_path": train["filename"].map(lambda x: str(data_root / "train_audio" / x)),
            "source": "focal",
            "source_id": train["filename"].astype(str),
            "start_seconds": np.nan,
            "end_seconds": np.nan,
            "labels": train["primary_label"].astype(str).map(lambda x: [x]),
        }
    )
    return rows


def make_soundscape_rows(soundscape: pd.DataFrame, data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    rows = pd.DataFrame(
        {
            "audio_path": soundscape["filename"].map(lambda x: str(data_root / "train_soundscapes" / x)),
            "source": "soundscape",
            "source_id": soundscape["filename"].astype(str),
            "start_seconds": soundscape["start"].map(parse_time_to_seconds),
            "end_seconds": soundscape["end"].map(parse_time_to_seconds),
            "labels": soundscape["primary_label"].map(parse_soundscape_labels),
        }
    )
    return rows


def soundscape_file_label_matrix(soundscape_rows: pd.DataFrame, label_space: LabelSpace) -> tuple[np.ndarray, np.ndarray]:
    grouped = soundscape_rows.groupby("source_id")["labels"].agg(lambda values: sorted({label for labels in values for label in labels}))
    files = grouped.index.to_numpy(dtype=str)
    targets = np.stack([multi_hot(labels, label_space) for labels in grouped.to_list()])
    return files, targets


def greedy_multilabel_group_split(
    files: np.ndarray,
    targets: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(files) < n_splits:
        raise ValueError(f"Cannot split {len(files)} soundscape files into {n_splits} splits")

    rng = np.random.default_rng(seed)
    positives = targets.sum(axis=0)
    rare_weight = 1.0 / np.maximum(positives, 1.0)
    sample_weights = (targets * rare_weight).sum(axis=1)
    sample_weights += 1e-6 * rng.random(len(files))
    order = np.argsort(-sample_weights)

    fold_targets = np.zeros((n_splits, targets.shape[1]), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    desired = targets.sum(axis=0) / n_splits
    max_fold_size = int(np.ceil(len(files) / n_splits))
    assignments = np.full(len(files), -1, dtype=np.int64)

    for idx in order:
        costs = []
        for fold_idx in range(n_splits):
            if fold_sizes[fold_idx] >= max_fold_size:
                costs.append(np.inf)
                continue
            candidate_targets = fold_targets[fold_idx] + targets[idx]
            label_cost = np.square((candidate_targets - desired) * rare_weight).sum()
            size_cost = np.square((fold_sizes[fold_idx] + 1) - (len(files) / n_splits))
            costs.append(label_cost + size_cost)
        best_fold = int(np.argmin(costs))
        assignments[idx] = best_fold
        fold_targets[best_fold] += targets[idx]
        fold_sizes[best_fold] += 1

    return assignments


def create_soundscape_split_plan(
    soundscape_rows: pd.DataFrame,
    label_space: LabelSpace,
    holdout_fraction: float,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("soundscape_holdout_fraction must be between 0 and 1")
    if n_folds < 2:
        raise ValueError("soundscape_cv_folds must be at least 2")

    files, targets = soundscape_file_label_matrix(soundscape_rows, label_space)
    holdout_splits = max(2, round(1.0 / holdout_fraction))
    holdout_assignments = greedy_multilabel_group_split(files, targets, holdout_splits, seed)
    holdout_mask = holdout_assignments == 0
    dev_files = files[~holdout_mask]
    dev_targets = targets[~holdout_mask]
    if len(dev_files) < n_folds:
        raise ValueError(f"Only {len(dev_files)} dev soundscape files remain for {n_folds} CV folds")

    cv_assignments = greedy_multilabel_group_split(dev_files, dev_targets, n_folds, seed + 1)
    rows = []
    for filename in files[holdout_mask]:
        rows.append({"filename": filename, "split": "holdout", "fold": -1})
    for filename, fold in zip(dev_files, cv_assignments):
        rows.append({"filename": filename, "split": "dev", "fold": int(fold)})
    return pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)


def load_or_create_soundscape_split_plan(
    soundscape_rows: pd.DataFrame,
    label_space: LabelSpace,
    plan_path: str | Path,
    holdout_fraction: float,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    plan_path = Path(plan_path)
    expected_files = set(soundscape_rows["source_id"].astype(str).unique())
    if plan_path.exists():
        plan = pd.read_csv(plan_path)
        plan["filename"] = plan["filename"].astype(str)
        plan["fold"] = plan["fold"].astype(int)
        plan_files = set(plan["filename"])
        if plan_files != expected_files:
            missing = sorted(expected_files - plan_files)[:5]
            extra = sorted(plan_files - expected_files)[:5]
            raise ValueError(
                "Existing soundscape split plan does not match soundscape files. "
                f"missing={missing}, extra={extra}. Delete or regenerate {plan_path} if this is intentional."
            )
        return plan

    plan = create_soundscape_split_plan(soundscape_rows, label_space, holdout_fraction, n_folds, seed)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_path, index=False)
    return plan


def make_split_from_plan(
    train: pd.DataFrame,
    soundscape: pd.DataFrame,
    data_root: str | Path,
    label_space: LabelSpace,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    focal_rows = make_focal_rows(train, data_root)
    soundscape_rows = make_soundscape_rows(soundscape, data_root)
    plan_path = cfg["soundscape_split_plan_path"]
    plan = load_or_create_soundscape_split_plan(
        soundscape_rows,
        label_space,
        plan_path,
        float(cfg["soundscape_holdout_fraction"]),
        int(cfg["soundscape_cv_folds"]),
        int(cfg["seed"]),
    )

    eval_split = str(cfg.get("soundscape_eval_split", "cv")).lower()
    fold = int(cfg.get("soundscape_fold", 0))
    holdout_files = set(plan.loc[plan["split"] == "holdout", "filename"].astype(str))
    dev_plan = plan[plan["split"] == "dev"]
    dev_files = set(dev_plan["filename"].astype(str))

    if eval_split == "cv":
        if fold < 0 or fold >= int(cfg["soundscape_cv_folds"]):
            raise ValueError(f"soundscape_fold must be in [0, {int(cfg['soundscape_cv_folds']) - 1}], got {fold}")
        val_files = set(dev_plan.loc[dev_plan["fold"] == fold, "filename"].astype(str))
        train_soundscape_files = dev_files - val_files
        split_fold: int | None = fold
    elif eval_split == "holdout":
        val_files = holdout_files
        train_soundscape_files = dev_files
        split_fold = None
    else:
        raise ValueError("soundscape_eval_split must be 'cv' or 'holdout'")

    overlap = train_soundscape_files & val_files
    if overlap:
        raise ValueError(f"Soundscape split leakage detected: {sorted(overlap)[:5]}")

    sound_train = soundscape_rows[soundscape_rows["source_id"].isin(train_soundscape_files)]
    sound_val = soundscape_rows[soundscape_rows["source_id"].isin(val_files)]
    mixed_train = pd.concat([focal_rows, sound_train], ignore_index=True)
    mixed_val = sound_val.reset_index(drop=True)
    return mixed_train, mixed_val, SplitInfo(
        eval_split=eval_split,
        fold=split_fold,
        plan_path=str(plan_path),
        train_soundscape_files=len(train_soundscape_files),
        val_soundscape_files=len(val_files),
    )


def make_mixed_split(
    train: pd.DataFrame,
    soundscape: pd.DataFrame,
    data_root: str | Path,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    focal_rows = make_focal_rows(train, data_root)
    soundscape_rows = make_soundscape_rows(soundscape, data_root)
    soundscape_files = np.array(sorted(soundscape_rows["source_id"].unique()))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(soundscape_files)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_files = set(shuffled[:val_count])
    train_files = set(shuffled[val_count:])
    sound_train = soundscape_rows[soundscape_rows["source_id"].isin(train_files)]
    sound_val = soundscape_rows[soundscape_rows["source_id"].isin(val_files)]
    mixed_train = pd.concat([focal_rows, sound_train], ignore_index=True)
    mixed_val = sound_val.reset_index(drop=True)
    return mixed_train, mixed_val


def attach_targets(rows: pd.DataFrame, label_space: LabelSpace) -> pd.DataFrame:
    rows = rows.copy()
    rows["target"] = rows["labels"].map(lambda labels: multi_hot(labels, label_space))
    return rows


def positive_class_weights(rows: pd.DataFrame, max_weight: float) -> np.ndarray:
    targets = np.stack(rows["target"].to_numpy())
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    weights = np.sqrt(negatives / np.maximum(positives, 1.0))
    weights[positives == 0] = 1.0
    return np.minimum(weights, max_weight).astype(np.float32)


def save_split(train_rows: pd.DataFrame, val_rows: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows.drop(columns=["target"], errors="ignore").to_json(out_dir / "train_rows.jsonl", orient="records", lines=True)
    val_rows.drop(columns=["target"], errors="ignore").to_json(out_dir / "val_rows.jsonl", orient="records", lines=True)
    pd.Series(sorted(train_rows.loc[train_rows["source"] == "soundscape", "source_id"].unique()), name="filename").to_csv(
        out_dir / "soundscape_train_files.csv", index=False
    )
    pd.Series(sorted(val_rows["source_id"].unique()), name="filename").to_csv(out_dir / "soundscape_val_files.csv", index=False)


def save_split_info(split_info: SplitInfo, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(
        {
            "eval_split": split_info.eval_split,
            "fold": "" if split_info.fold is None else split_info.fold,
            "plan_path": split_info.plan_path,
            "train_soundscape_files": split_info.train_soundscape_files,
            "val_soundscape_files": split_info.val_soundscape_files,
        }
    ).to_csv(out_dir / "split_info.csv", header=False)
