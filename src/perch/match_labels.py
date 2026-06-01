from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import load_config


def normalize_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def binomial_name(value: str) -> str:
    parts = normalize_name(value).split()
    if len(parts) < 2:
        return ""
    return " ".join(parts[:2])


def load_perch_labels(model_path: str | Path) -> pd.DataFrame:
    assets = Path(model_path) / "assets"
    labels = pd.read_csv(assets / "labels.csv", header=None, names=["perch_label"])
    ebird = pd.read_csv(assets / "perch_v2_ebird_classes.csv", header=None, names=["perch_ebird_code"])
    if len(labels) != len(ebird):
        raise ValueError(f"Perch label files have different lengths: {len(labels)} vs {len(ebird)}")

    labels["perch_index"] = labels.index
    labels["perch_ebird_code"] = ebird["perch_ebird_code"]
    labels["normalized_scientific_name"] = labels["perch_label"].map(normalize_name)
    labels["binomial_scientific_name"] = labels["perch_label"].map(binomial_name)
    return labels


def unique_lookup(df: pd.DataFrame, key: str) -> dict[str, pd.Series]:
    counts = df[key].value_counts()
    unique_keys = set(counts[counts == 1].index)
    return {row[key]: row for _, row in df[df[key].isin(unique_keys)].iterrows() if row[key]}


def match_labels(taxonomy: pd.DataFrame, perch_labels: pd.DataFrame) -> pd.DataFrame:
    exact_lookup = unique_lookup(perch_labels, "normalized_scientific_name")
    binomial_lookup = unique_lookup(perch_labels, "binomial_scientific_name")

    matches = []
    for row in taxonomy.itertuples(index=False):
        normalized = normalize_name(row.scientific_name)
        binomial = binomial_name(row.scientific_name)
        match_type = "unmatched"
        perch = None

        if normalized in exact_lookup:
            perch = exact_lookup[normalized]
            match_type = "exact_scientific_name"
        elif binomial in binomial_lookup:
            perch = binomial_lookup[binomial]
            match_type = "binomial_scientific_name"

        matches.append(
            {
                "primary_label": row.primary_label,
                "scientific_name": row.scientific_name,
                "common_name": row.common_name,
                "class_name": row.class_name,
                "match_type": match_type,
                "perch_index": None if perch is None else int(perch["perch_index"]),
                "perch_label": None if perch is None else perch["perch_label"],
                "perch_ebird_code": None if perch is None else perch["perch_ebird_code"],
            }
        )

    return pd.DataFrame(matches)


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    taxonomy = pd.read_csv(Path(cfg["data_root"]) / "taxonomy.csv")
    perch_labels = load_perch_labels(cfg["perch_model_path"])
    matches = match_labels(taxonomy, perch_labels)

    out_dir = PROJECT_ROOT / "data" / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "birdclef_to_perch_label_mapping.csv"
    matches.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")
    print("Match counts")
    print(matches["match_type"].value_counts().to_string())
    print()
    print("Unmatched labels")
    unmatched = matches[matches["match_type"] == "unmatched"]
    if unmatched.empty:
        print("none")
    else:
        print(unmatched[["primary_label", "scientific_name", "common_name", "class_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
