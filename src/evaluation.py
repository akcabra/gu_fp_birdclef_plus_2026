from __future__ import annotations

import pandas as pd

from src.metrics import per_class_auc


def labels_in_rows(rows) -> set[str]:
    labels = set()
    for row_labels in rows["labels"]:
        labels.update(str(label) for label in row_labels)
    return labels


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
    print()
    print("Validation group summary")
    out = summary.copy()
    out["mean_auc"] = out["mean_auc"].map(lambda value: "n/a" if pd.isna(value) else f"{value:.5f}")
    out["median_auc"] = out["median_auc"].map(lambda value: "n/a" if pd.isna(value) else f"{value:.5f}")
    print(out.to_string(index=False))
