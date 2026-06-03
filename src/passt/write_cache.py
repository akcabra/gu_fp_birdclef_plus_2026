from __future__ import annotations

from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import attach_targets, build_label_space, load_tables, make_split_from_plan
from src.passt.cache import write_passt_cache


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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


def main() -> None:
    start = perf_counter()
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    train_csv, soundscape_csv, _, sample_submission = load_tables(cfg["data_root"])
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

    train_cache_rows = expand_focal_rows(train_rows, cfg["passt_cache_focal_crops_per_recording"])
    val_cache_rows = make_passt_val_rows(val_rows, validation_crop_offsets(cfg))
    input_samples = int(32000 * cfg["passt_input_seconds"])

    print(
        "Split: "
        f"eval_split={split_info.eval_split}, "
        f"fold={'n/a' if split_info.fold is None else split_info.fold}, "
        f"train_soundscape_files={split_info.train_soundscape_files}, "
        f"val_soundscape_files={split_info.val_soundscape_files}"
    )
    print(f"Split plan: {split_info.plan_path}")
    print()

    print(f"Writing PaSST train embedding cache to {cfg['passt_train_cache_path']}")
    write_passt_cache(
        train_cache_rows,
        cfg["passt_train_cache_path"],
        batch_size=cfg["passt_batch_size"],
        device=cfg["passt_device"],
        arch=cfg["passt_arch"],
        include_logits=cfg["passt_cache_include_logits"],
        input_samples=input_samples,
    )

    print(f"Writing PaSST validation embedding cache to {cfg['passt_val_cache_path']}")
    write_passt_cache(
        val_cache_rows,
        cfg["passt_val_cache_path"],
        batch_size=cfg["passt_batch_size"],
        device=cfg["passt_device"],
        arch=cfg["passt_arch"],
        include_logits=cfg["passt_cache_include_logits"],
        input_samples=input_samples,
    )
    print(f"Finished writing PaSST caches in {format_duration(perf_counter() - start)}")


if __name__ == "__main__":
    main()
