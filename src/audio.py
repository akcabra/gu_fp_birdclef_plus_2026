from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

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


def make_tf_dataset(rows, batch_size: int, training: bool):
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
