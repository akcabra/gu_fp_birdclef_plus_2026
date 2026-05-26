from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.metrics import roc_auc_score


class ParticipantVisibleError(Exception):
    pass


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


# Official BirdCLEF+ macro ROC-AUC behavior for local validation.
def challenge_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    solution = solution.copy()
    submission = submission.copy()
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if not pandas.api.types.is_numeric_dtype(submission.values):
        bad_dtypes = {
            column: submission[column].dtype
            for column in submission.columns
            if not pandas.api.types.is_numeric_dtype(submission[column])
        }
        raise ParticipantVisibleError(f"Invalid submission data types found: {bad_dtypes}")

    solution_sums = solution.sum(axis=0)
    scored_columns = list(solution_sums[solution_sums > 0].index.values)
    assert len(scored_columns) > 0

    return roc_auc_score(solution[scored_columns].values, submission[scored_columns].values, average="macro")


def per_class_auc(y_true: np.ndarray, y_score: np.ndarray, labels: list[str]) -> pd.DataFrame:
    rows = []
    for idx, label in enumerate(labels):
        positives = int(y_true[:, idx].sum())
        negatives = int(len(y_true) - positives)
        if positives == 0 or negatives == 0:
            auc = np.nan
        else:
            auc = roc_auc_score(y_true[:, idx], y_score[:, idx])
        rows.append({"label": label, "auc": auc, "positives": positives, "negatives": negatives})
    return pd.DataFrame(rows)


def challenge_score_from_arrays(y_true: np.ndarray, y_score: np.ndarray, labels: list[str]) -> float:
    solution = pd.DataFrame(y_true, columns=labels)
    submission = pd.DataFrame(y_score, columns=labels)
    solution.insert(0, "row_id", np.arange(len(solution)))
    submission.insert(0, "row_id", np.arange(len(submission)))
    return float(challenge_score(solution, submission, "row_id"))


def rare_common_summary(per_class: pd.DataFrame, train_rows: pd.DataFrame) -> pd.DataFrame:
    counts = {}
    for labels in train_rows["labels"]:
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    out = per_class.copy()
    out["train_count"] = out["label"].map(counts).fillna(0).astype(int)
    bins = [-1, 5, 20, 100, np.inf]
    names = ["0-5", "6-20", "21-100", "100+"]
    out["count_bin"] = pd.cut(out["train_count"], bins=bins, labels=names)
    return out.groupby("count_bin", observed=True)["auc"].agg(["count", "mean", "median"]).reset_index()
