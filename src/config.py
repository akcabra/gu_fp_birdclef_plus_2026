from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["project_root"] = str(PROJECT_ROOT)
    for key in ["perch_model_path", "perch_cpu_model_path", "perch_label_mapping_path"]:
        if key in cfg:
            cfg[key] = str(_resolve_project_path(cfg[key]))
    for key in ["load_model_weights_path", "save_model_weights_path", "best_model_weights_path"]:
        if key in cfg and cfg[key]:
            cfg[key] = str(_resolve_project_path(cfg[key]))
    for key in ["train_embedding_cache_path", "val_embedding_cache_path"]:
        if key in cfg:
            cfg[key] = str(_resolve_project_path(cfg[key]))
    for key in [
        "passt_train_cache_path",
        "passt_val_cache_path",
        "passt_load_model_weights_path",
        "passt_save_model_weights_path",
        "passt_best_model_weights_path",
        "perch_val_predictions_path",
        "passt_val_predictions_path",
        "ensemble_val_predictions_path",
        "ensemble_classwise_alphas_path",
    ]:
        if key in cfg and cfg[key]:
            cfg[key] = str(_resolve_project_path(cfg[key]))
    for key in ["data_root"]:
        cfg[key] = str(Path(cfg[key]).expanduser())
    return cfg


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
