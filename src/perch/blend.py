from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import sigmoid


class PerchScoreBlender:
    def __init__(self, labels: list[str], mapping_path: str | Path, alpha: float):
        self.labels = labels
        self.alpha = alpha

        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        mapping = pd.read_csv(mapping_path)
        mapping = mapping[mapping["perch_index"].notna()].copy()
        mapping["birdclef_index"] = mapping["primary_label"].astype(str).map(label_to_idx)
        mapping = mapping[mapping["birdclef_index"].notna()].copy()

        self.birdclef_indices = mapping["birdclef_index"].astype(int).to_numpy()
        self.perch_indices = mapping["perch_index"].astype(int).to_numpy()

    @property
    def matched_count(self) -> int:
        return len(self.birdclef_indices)

    def map_scores(self, perch_label_output: np.ndarray) -> np.ndarray:
        perch_scores = perch_label_output
        if perch_scores.min() < 0.0 or perch_scores.max() > 1.0:
            perch_scores = sigmoid(perch_scores)

        mapped = np.zeros((len(perch_scores), len(self.labels)), dtype=np.float32)
        mapped[:, self.birdclef_indices] = perch_scores[:, self.perch_indices]
        return mapped

    def blend_mapped(self, head_scores: np.ndarray, mapped_perch_scores: np.ndarray) -> np.ndarray:
        blended = head_scores.copy()
        blended[:, self.birdclef_indices] = (
            self.alpha * head_scores[:, self.birdclef_indices]
            + (1.0 - self.alpha) * mapped_perch_scores[:, self.birdclef_indices]
        )
        return blended

    def blend(self, head_scores: np.ndarray, perch_label_output: np.ndarray) -> np.ndarray:
        return self.blend_mapped(head_scores, self.map_scores(perch_label_output))
