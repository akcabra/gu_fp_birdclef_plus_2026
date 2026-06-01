from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

def crop_or_pad(
    audio: np.ndarray,
    start_seconds: float | None = None,
    random_crop: bool = False,
) -> np.ndarray:
    if start_seconds is not None and not np.isnan(start_seconds):
        start = int(round(start_seconds * 32000))
    elif random_crop and len(audio) > 160000:
        start = np.random.randint(0, len(audio) - 160000 + 1)
    else:
        start = max(0, (len(audio) - 160000) // 2)
    start = min(max(0, start), max(0, len(audio) - 160000))

    clip = audio[start : start + 160000]
    if len(clip) < 160000:
        clip = np.pad(clip, (0, 160000 - len(clip)))
    return clip.astype(np.float32)


def load_clip_np(
    path: str,
    source: str,
    start_seconds: float,
    training: bool,
) -> np.ndarray:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    start = None if source == "focal" else start_seconds
    random_crop = training and source == "focal"
    return crop_or_pad(audio, start_seconds=start, random_crop=random_crop)


def make_tf_dataset(rows, batch_size: int, training: bool, sample_weights: np.ndarray | None = None, seed: int = 42):
    import tensorflow as tf

    paths = rows["audio_path"].astype(str).to_numpy()
    sources = rows["source"].astype(str).to_numpy()
    starts = rows["start_seconds"].fillna(-1).astype(np.float32).to_numpy()
    targets = np.stack(rows["target"].to_numpy()).astype(np.float32)

    if training and sample_weights is not None:
        log_probs = tf.math.log(tf.constant(sample_weights / sample_weights.sum(), dtype=tf.float32))[tf.newaxis, :]
        paths = tf.constant(paths)
        sources = tf.constant(sources)
        starts = tf.constant(starts)
        targets = tf.constant(targets)

        def _sample(_):
            index = tf.random.categorical(log_probs, 1, seed=seed)[0, 0]
            return tf.gather(paths, index), tf.gather(sources, index), tf.gather(starts, index), tf.gather(targets, index)

        ds = tf.data.Dataset.range(len(targets)).map(_sample, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, sources, starts, targets))

    if training and sample_weights is None:
        ds = ds.shuffle(min(len(rows), 4096), reshuffle_each_iteration=True)

    def _load(path, source, start, target):
        waveform = tf.numpy_function(
            func=lambda p, s, st: load_clip_np(
                p.decode("utf-8"),
                s.decode("utf-8"),
                float(st),
                training,
            ),
            inp=[path, source, start],
            Tout=tf.float32,
        )
        waveform.set_shape([160000])
        target.set_shape([targets.shape[1]])
        return waveform, target

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_multicrop_tf_dataset(rows, batch_size: int, offsets_seconds: list[float]):
    import tensorflow as tf

    paths = []
    sources = []
    starts = []
    targets = []
    row_indices = []

    for row_idx, row in enumerate(rows.itertuples(index=False)):
        for offset in offsets_seconds:
            paths.append(str(row.audio_path))
            sources.append(str(row.source))
            starts.append(float(row.start_seconds) + offset)
            targets.append(row.target)
            row_indices.append(row_idx)

    paths = np.asarray(paths, dtype=object)
    sources = np.asarray(sources, dtype=object)
    starts = np.asarray(starts, dtype=np.float32)
    targets = np.stack(targets).astype(np.float32)
    row_indices = np.asarray(row_indices, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, sources, starts, targets))

    def _load(path, source, start, target):
        waveform = tf.numpy_function(
            func=lambda p, s, st: load_clip_np(
                p.decode("utf-8"),
                s.decode("utf-8"),
                float(st),
                False,
            ),
            inp=[path, source, start],
            Tout=tf.float32,
        )
        waveform.set_shape([160000])
        target.set_shape([targets.shape[1]])
        return waveform, target

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, row_indices
