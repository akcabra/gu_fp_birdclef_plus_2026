from __future__ import annotations

from pathlib import Path

import numpy as np

from src.audio import load_clip_np
from src.passt.extractor import PaSSTExtractor


def _batched_rows(rows, batch_size: int):
    batch_waveforms = []
    batch_targets = []
    batch_row_indices = []
    batch_source_row_indices = []
    has_row_indices = "row_index" in rows.columns
    has_source_indices = "source_row_index" in rows.columns

    for row_idx, row in enumerate(rows.itertuples(index=False)):
        batch_waveforms.append(
            load_clip_np(
                str(row.audio_path),
                str(row.source),
                float(row.start_seconds) if not np.isnan(row.start_seconds) else -1.0,
                training=str(row.source) == "focal",
            )
        )
        batch_targets.append(row.target)
        if has_row_indices:
            batch_row_indices.append(int(row.row_index))
        if has_source_indices:
            batch_source_row_indices.append(int(row.source_row_index))

        if len(batch_waveforms) == batch_size:
            yield batch_waveforms, batch_targets, batch_row_indices, batch_source_row_indices
            batch_waveforms = []
            batch_targets = []
            batch_row_indices = []
            batch_source_row_indices = []

    if batch_waveforms:
        yield batch_waveforms, batch_targets, batch_row_indices, batch_source_row_indices


def write_passt_cache(
    rows,
    path: str | Path,
    batch_size: int,
    device: str = "auto",
    arch: str = "",
    include_logits: bool = False,
    input_samples: int = 320000,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extractor = PaSSTExtractor(device=device, arch=arch, include_logits=include_logits)

    embeddings = []
    logits = []
    targets = []
    row_indices = []
    source_row_indices = []

    for batch_waveforms, batch_targets, batch_row_indices, batch_source_row_indices in _batched_rows(rows, batch_size):
        waveforms = np.stack(batch_waveforms).astype(np.float32)
        if waveforms.shape[1] < input_samples:
            waveforms = np.pad(waveforms, ((0, 0), (0, input_samples - waveforms.shape[1])))
        elif waveforms.shape[1] > input_samples:
            waveforms = waveforms[:, :input_samples]
        output = extractor.extract(waveforms)
        embeddings.append(output.embeddings)
        if output.logits is not None:
            logits.append(output.logits)
        targets.append(np.stack(batch_targets).astype(np.float32))
        row_indices.extend(batch_row_indices)
        source_row_indices.extend(batch_source_row_indices)

    arrays = {
        "embeddings": np.concatenate(embeddings, axis=0).astype(np.float32),
        "targets": np.concatenate(targets, axis=0).astype(np.float32),
    }
    if logits:
        arrays["passt_logits"] = np.concatenate(logits, axis=0).astype(np.float32)
    if row_indices:
        arrays["row_indices"] = np.asarray(row_indices, dtype=np.int32)
    if source_row_indices:
        arrays["source_row_indices"] = np.asarray(source_row_indices, dtype=np.int32)
    np.savez_compressed(path, **arrays)


def load_passt_cache(path: str | Path):
    return np.load(path)
