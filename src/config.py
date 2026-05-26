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
    for key in ["perch_model_path", "perch_cpu_model_path"]:
        cfg[key] = str(_resolve_project_path(cfg[key]))
    for key in ["data_root"]:
        cfg[key] = str(Path(cfg[key]).expanduser())
    return cfg


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
