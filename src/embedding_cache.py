from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def write_embedding_cache(raw_model: tf.keras.Model, ds, path: str | Path, perch_mapper=None, row_indices=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    perch_layer = raw_model.get_layer("perch")

    embeddings = []
    mapped_perch_scores = []
    targets = []
    for x_batch, y_batch in ds:
        outputs = perch_layer.saved_model_layer(x_batch, training=False)
        embeddings.append(outputs["embedding"].numpy().astype(np.float32))
        if perch_mapper is None:
            mapped_perch_scores.append(np.zeros((len(x_batch), 234), dtype=np.float32))
        else:
            mapped_perch_scores.append(perch_mapper.map_scores(outputs["label"].numpy()))
        targets.append(y_batch.numpy().astype(np.float32))

    arrays = {
        "embeddings": np.concatenate(embeddings, axis=0),
        "mapped_perch_scores": np.concatenate(mapped_perch_scores, axis=0),
        "targets": np.concatenate(targets, axis=0),
    }
    if row_indices is not None:
        arrays["row_indices"] = row_indices.astype(np.int32)
    np.savez_compressed(path, **arrays)


def load_embedding_cache(path: str | Path):
    return np.load(path)


def make_embedding_dataset(cache, batch_size: int, training: bool):
    embeddings = cache["embeddings"].astype(np.float32)
    mapped_perch_scores = cache["mapped_perch_scores"].astype(np.float32)
    targets = cache["targets"].astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices(((embeddings, mapped_perch_scores), targets))
    if training:
        ds = ds.shuffle(min(len(targets), 4096), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
