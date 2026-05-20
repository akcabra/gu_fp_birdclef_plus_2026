from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
