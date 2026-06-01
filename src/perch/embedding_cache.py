from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def write_embedding_cache(
    raw_model: tf.keras.Model,
    ds,
    path: str | Path,
    perch_mapper=None,
    row_indices=None,
    source_row_indices=None,
) -> None:
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
    if source_row_indices is not None:
        arrays["source_row_indices"] = source_row_indices.astype(np.int32)
    np.savez_compressed(path, **arrays)


def load_embedding_cache(path: str | Path):
    return np.load(path)


def make_embedding_dataset(cache, batch_size: int, training: bool, sample_weights: np.ndarray | None = None, seed: int = 42):
    embeddings = cache["embeddings"].astype(np.float32)
    mapped_perch_scores = cache["mapped_perch_scores"].astype(np.float32)
    targets = cache["targets"].astype(np.float32)

    if training and "source_row_indices" in cache:
        source_row_indices = cache["source_row_indices"].astype(np.int32)
        group_ids = np.unique(source_row_indices)
        group_to_output = {group_id: idx for idx, group_id in enumerate(group_ids)}
        group_indices = np.array([group_to_output[group_id] for group_id in source_row_indices], dtype=np.int32)
        group_counts = np.bincount(group_indices)
        max_group_count = int(group_counts.max())
        candidate_indices = np.zeros((len(group_ids), max_group_count), dtype=np.int32)
        for group_idx in range(len(group_ids)):
            entries = np.flatnonzero(group_indices == group_idx)
            candidate_indices[group_idx, : len(entries)] = entries

        if sample_weights is None:
            group_weights = np.ones(len(group_ids), dtype=np.float64)
        else:
            group_weights = sample_weights

        group_log_probs = tf.math.log(tf.constant(group_weights / group_weights.sum(), dtype=tf.float32))[tf.newaxis, :]
        embeddings = tf.constant(embeddings)
        mapped_perch_scores = tf.constant(mapped_perch_scores)
        targets = tf.constant(targets)
        candidate_indices = tf.constant(candidate_indices)
        group_counts = tf.constant(group_counts, dtype=tf.int64)

        def sample_example(_):
            if sample_weights is None:
                group_index = _
            else:
                group_index = tf.random.categorical(group_log_probs, 1, seed=seed)[0, 0]
            crop_offset = tf.random.uniform((), maxval=tf.gather(group_counts, group_index), dtype=tf.int64, seed=seed)
            index = tf.gather_nd(candidate_indices, [[group_index, crop_offset]])[0]
            return (tf.gather(embeddings, index), tf.gather(mapped_perch_scores, index)), tf.gather(targets, index)

        ds = tf.data.Dataset.range(len(group_ids))
        if sample_weights is None:
            ds = ds.shuffle(min(len(group_ids), 4096), reshuffle_each_iteration=True)
        ds = ds.map(sample_example, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds = tf.data.Dataset.from_tensor_slices(((embeddings, mapped_perch_scores), targets))
    if training:
        ds = ds.shuffle(min(len(targets), 4096), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
