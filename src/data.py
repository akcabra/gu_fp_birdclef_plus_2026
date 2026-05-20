from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class LabelSpace:
    labels: list[str]
    label_to_idx: dict[str, int]
    idx_to_label: dict[int, str]


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
    train_files, val_files = train_test_split(
        soundscape_files,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    sound_train = soundscape_rows[soundscape_rows["source_id"].isin(train_files)]
    sound_val = soundscape_rows[soundscape_rows["source_id"].isin(val_files)]

    mixed_train = pd.concat([focal_rows, sound_train], ignore_index=True)
    mixed_val = sound_val.reset_index(drop=True)
    return mixed_train, mixed_val


def attach_targets(rows: pd.DataFrame, label_space: LabelSpace) -> pd.DataFrame:
    rows = rows.copy()
    rows["target"] = rows["labels"].map(lambda labels: multi_hot(labels, label_space))
    return rows


def save_split(train_rows: pd.DataFrame, val_rows: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows.drop(columns=["target"], errors="ignore").to_json(out_dir / "train_rows.jsonl", orient="records", lines=True)
    val_rows.drop(columns=["target"], errors="ignore").to_json(out_dir / "val_rows.jsonl", orient="records", lines=True)
    pd.Series(sorted(train_rows.loc[train_rows["source"] == "soundscape", "source_id"].unique()), name="filename").to_csv(
        out_dir / "soundscape_train_files.csv", index=False
    )
    pd.Series(sorted(val_rows["source_id"].unique()), name="filename").to_csv(out_dir / "soundscape_val_files.csv", index=False)
