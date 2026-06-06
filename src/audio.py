from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 32000
CLIP_SAMPLES = 160000


def energy_biased_start(
    audio: np.ndarray,
    clip_samples: int = CLIP_SAMPLES,
    hop_samples: int = TARGET_SAMPLE_RATE // 2,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> int:
    if len(audio) <= clip_samples:
        return 0

    hop_samples = max(1, int(hop_samples))
    starts = np.arange(0, len(audio) - clip_samples + 1, hop_samples, dtype=np.int64)
    last_start = len(audio) - clip_samples
    if starts[-1] != last_start:
        starts = np.append(starts, last_start)

    cumsum = np.concatenate([[0.0], np.cumsum(np.square(audio.astype(np.float64, copy=False)))])
    energies = (cumsum[starts + clip_samples] - cumsum[starts]) / clip_samples
    weights = np.power(np.maximum(energies, eps), temperature)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return int(np.random.randint(0, last_start + 1))

    probs = weights / weight_sum
    coarse_start = int(np.random.choice(starts, p=probs))
    jitter_limit = min(hop_samples, last_start - coarse_start)
    if jitter_limit > 0:
        coarse_start += int(np.random.randint(0, jitter_limit + 1))
    return min(max(0, coarse_start), last_start)


def crop_or_pad(
    audio: np.ndarray,
    start_seconds: float | None = None,
    crop_mode: str = "center",
) -> np.ndarray:
    if start_seconds is not None and not np.isnan(start_seconds):
        start = int(round(start_seconds * TARGET_SAMPLE_RATE))
    elif crop_mode == "random":
        if len(audio) > CLIP_SAMPLES:
            start = np.random.randint(0, len(audio) - CLIP_SAMPLES + 1)
        else:
            start = 0
    elif crop_mode == "energy_biased":
        start = energy_biased_start(audio)
    elif crop_mode == "center":
        start = max(0, (len(audio) - CLIP_SAMPLES) // 2)
    else:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")
    start = min(max(0, start), max(0, len(audio) - CLIP_SAMPLES))

    clip = audio[start : start + CLIP_SAMPLES]
    if len(clip) < CLIP_SAMPLES:
        clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
    return clip.astype(np.float32)


def load_clip_np(
    path: str,
    source: str,
    start_seconds: float,
    training: bool,
    augmentation: str = "random_crop",
) -> np.ndarray:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    start = None if source == "focal" else start_seconds
    if training and source == "focal":
        if augmentation == "random_crop":
            crop_mode = "random"
        elif augmentation == "energy_biased_crop":
            crop_mode = "energy_biased"
        elif augmentation in {"center_crop", "none"}:
            crop_mode = "center"
        else:
            raise ValueError(f"Unknown augmentation: {augmentation}")
    else:
        crop_mode = "center"
    return crop_or_pad(audio, start_seconds=start, crop_mode=crop_mode)


def make_tf_dataset(
    rows,
    batch_size: int,
    training: bool,
    sample_weights: np.ndarray | None = None,
    seed: int = 42,
    augmentation: str = "random_crop",
    shuffle: bool = True,
):
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

    if training and sample_weights is None and shuffle:
        ds = ds.shuffle(min(len(rows), 4096), reshuffle_each_iteration=True)

    def _load(path, source, start, target):
        waveform = tf.numpy_function(
            func=lambda p, s, st: load_clip_np(
                p.decode("utf-8"),
                s.decode("utf-8"),
                float(st),
                training,
                augmentation,
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
