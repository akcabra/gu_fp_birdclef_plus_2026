#!/usr/bin/env python3
"""Create dataset distribution figures for the final report.

The script avoids plotting-library dependencies. It writes a PDF figure and
the CSV file with the plotted counts.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


TAXONOMY_ORDER = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia", "Unknown"]
SERIES = [
    ("target_classes", "Target classes"),
    ("focal_recordings", "Focal recordings"),
    ("soundscape_positives", "Soundscape positives"),
]
COLORS = {
    "target_classes": "#3B6FB6",
    "focal_recordings": "#2D8A5D",
    "soundscape_positives": "#C45A36",
}
STALE_OUTPUTS = [
    "data_taxonomy_distribution.svg",
    "data_taxonomy_distribution_pgfplots.tex",
    "data_focal_class_counts.csv",
    "data_focal_class_imbalance.svg",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_taxonomy_map(taxonomy_rows: list[dict[str, str]]) -> dict[str, str]:
    mapping = {}
    for row in taxonomy_rows:
        label = str(row["primary_label"])
        group = row.get("class_name") or "Unknown"
        mapping[label] = group
    return mapping


def target_labels(sample_submission_path: Path) -> list[str]:
    with sample_submission_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [name for name in header if name != "row_id"]


def sorted_groups(groups: set[str]) -> list[str]:
    ordered = [group for group in TAXONOMY_ORDER if group in groups]
    ordered.extend(sorted(groups - set(ordered)))
    return ordered


def compute_taxonomy_counts(data_root: Path) -> list[dict[str, object]]:
    taxonomy_rows = read_csv(data_root / "taxonomy.csv")
    train_rows = read_csv(data_root / "train.csv")
    soundscape_rows = read_csv(data_root / "train_soundscapes_labels.csv")
    label_to_group = label_taxonomy_map(taxonomy_rows)

    target_counter = Counter()
    for label in target_labels(data_root / "sample_submission.csv"):
        target_counter[label_to_group.get(label, "Unknown")] += 1

    focal_counter = Counter()
    for row in train_rows:
        label = str(row["primary_label"])
        group = label_to_group.get(label) or row.get("class_name") or "Unknown"
        focal_counter[group] += 1

    soundscape_counter = Counter()
    for row in soundscape_rows:
        labels = [label for label in str(row["primary_label"]).split(";") if label]
        for label in labels:
            soundscape_counter[label_to_group.get(label, "Unknown")] += 1

    groups = sorted_groups(set(target_counter) | set(focal_counter) | set(soundscape_counter))
    return [
        {
            "taxonomy_group": group,
            "target_classes": target_counter[group],
            "focal_recordings": focal_counter[group],
            "soundscape_positives": soundscape_counter[group],
        }
        for group in groups
    ]


def nice_int(value: int | float) -> str:
    return f"{int(value):,}"


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))


def pdf_escape(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_text(x: float, y: float, text: object, size: int = 9, font: str = "F1") -> str:
    return f"BT /{font} {size} Tf {x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET\n"


def pdf_fill_color(color: str) -> str:
    r, g, b = hex_to_rgb(color)
    return f"{r:.4f} {g:.4f} {b:.4f} rg\n"


def write_pdf(path: Path, width: int, height: int, content: str) -> None:
    stream = content.encode("utf-8")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> /Contents 6 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(pdf)


def taxonomy_pdf_content(rows: list[dict[str, object]], width: int, height: int) -> str:
    margin_x = 28
    panel_gap = 20
    panel_width = (width - 2 * margin_x - panel_gap * (len(SERIES) - 1)) / len(SERIES)
    label_width = 60
    value_space = 35
    bar_width = panel_width - label_width - value_space
    row_height = 25
    top_y = height - 24

    commands = [
        "1 1 1 rg\n0 0 {0} {1} re f\n".format(width, height),
    ]

    for panel_idx, (key, title) in enumerate(SERIES):
        x0 = margin_x + panel_idx * (panel_width + panel_gap)
        max_value = max(int(row[key]) for row in rows) or 1
        commands.extend(
            [
                pdf_fill_color("#222222"),
                pdf_text(x0, height - 26, title, 10, "F2"),
            ]
        )

        for row_idx, row in enumerate(rows):
            y = top_y - 32 - row_idx * row_height
            group = str(row["taxonomy_group"])
            value = int(row[key])
            current_bar_width = max(1, bar_width * value / max_value)
            commands.extend(
                [
                    pdf_fill_color("#222222"),
                    pdf_text(x0, y + 4, group, 8, "F1"),
                    pdf_fill_color(COLORS[key]),
                    f"{x0 + label_width:.2f} {y:.2f} {current_bar_width:.2f} 12 re f\n",
                    pdf_fill_color("#333333"),
                    pdf_text(x0 + label_width + current_bar_width + 4, y + 3, nice_int(value), 7, "F1"),
                ]
            )
    return "".join(commands)


def write_taxonomy_pdf(path: Path, rows: list[dict[str, object]]) -> None:
    width = 720
    height = 190
    write_pdf(path, width, height, taxonomy_pdf_content(rows, width, height))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / ".cache/kagglehub/competitions/birdclef-2026",
        help="BirdCLEF+ 2026 dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where the PDF figure and count CSV are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_counts = compute_taxonomy_counts(args.data_root)

    write_csv(
        output_dir / "data_taxonomy_distribution_counts.csv",
        taxonomy_counts,
        ["taxonomy_group", "target_classes", "focal_recordings", "soundscape_positives"],
    )

    for filename in STALE_OUTPUTS:
        stale_path = output_dir / filename
        if stale_path.exists():
            stale_path.unlink()

    write_taxonomy_pdf(output_dir / "data_taxonomy_distribution.pdf", taxonomy_counts)

    print(f"Wrote dataset figure to {output_dir}")
    print(f"Taxonomy counts: {output_dir / 'data_taxonomy_distribution_counts.csv'}")
    print(f"PDF figure: {output_dir / 'data_taxonomy_distribution.pdf'}")


if __name__ == "__main__":
    main()
