from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

def crop_or_pad(
    audio: np.ndarray,
    sample_rate: int,
    clip_seconds: float,
    start_seconds: float | None = None,
    random_crop: bool = False,
) -> np.ndarray:
    n_samples = int(sample_rate * clip_seconds)
    if start_seconds is not None and not np.isnan(start_seconds):
        start = int(round(start_seconds * sample_rate))
    elif random_crop and len(audio) > n_samples:
        start = np.random.randint(0, len(audio) - n_samples + 1)
    else:
        start = max(0, (len(audio) - n_samples) // 2)

    clip = audio[start : start + n_samples]
    if len(clip) < n_samples:
        clip = np.pad(clip, (0, n_samples - len(clip)))
    return clip.astype(np.float32)


def load_clip_np(
    path: str,
    source: str,
    start_seconds: float,
    sample_rate: int,
    clip_seconds: float,
    training: bool,
) -> np.ndarray:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    start = None if source == "focal" else start_seconds
    random_crop = training and source == "focal"
    return crop_or_pad(audio, sample_rate, clip_seconds, start_seconds=start, random_crop=random_crop)


def make_tf_dataset(rows, sample_rate: int, clip_seconds: float, batch_size: int, training: bool):
    paths = rows["audio_path"].astype(str).to_numpy()
    sources = rows["source"].astype(str).to_numpy()
    starts = rows["start_seconds"].fillna(-1).astype(np.float32).to_numpy()
    targets = np.stack(rows["target"].to_numpy()).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, sources, starts, targets))
    if training:
        ds = ds.shuffle(min(len(rows), 4096), reshuffle_each_iteration=True)

    def _load(path, source, start, target):
        waveform = tf.numpy_function(
            func=lambda p, s, st: load_clip_np(
                p.decode("utf-8"),
                s.decode("utf-8"),
                float(st),
                sample_rate,
                clip_seconds,
                training,
            ),
            inp=[path, source, start],
            Tout=tf.float32,
        )
        waveform.set_shape([int(sample_rate * clip_seconds)])
        target.set_shape([targets.shape[1]])
        return waveform, target

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
