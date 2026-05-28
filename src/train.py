from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf

from .metrics import challenge_score_from_arrays, sigmoid
from .model import set_backbone_trainable


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class EpochTimingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = perf_counter() - self.epoch_start
        print(f" - epoch_time: {format_duration(elapsed)}")


class ChallengeScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, labels: list[str]):
        super().__init__()
        self.val_ds = val_ds
        self.labels = labels
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true, logits = predict_dataset(self.model, self.val_ds)
        value = challenge_score_from_arrays(y_true, sigmoid(logits), self.labels)
        logs["val_challenge_score"] = value
        self.best = max(self.best, value)
        print(f" - val_challenge_score: {value:.5f}")


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )


def train_head_only(model, train_ds, val_ds, labels, cfg):
    print("Starting head-only training")
    start = perf_counter()
    set_backbone_trainable(model, False)
    compile_model(model, cfg["head_lr"])
    checkpoint_dir = Path("outputs") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EpochTimingCallback(),
        ChallengeScoreCallback(val_ds, labels),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_dir / "best_head_only.weights.h5",
            monitor="val_challenge_score",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs_head"],
        callbacks=callbacks,
        verbose=2,
    )
    print(f"Finished head-only training in {format_duration(perf_counter() - start)}")
    return history


def train_finetune(model, train_ds, val_ds, labels, cfg):
    print("Starting Perch fine-tuning")
    start = perf_counter()
    set_backbone_trainable(model, True)
    compile_model(model, cfg["backbone_lr"])
    checkpoint_dir = Path("outputs") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EpochTimingCallback(),
        ChallengeScoreCallback(val_ds, labels),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_dir / "best_finetuned.weights.h5",
            monitor="val_challenge_score",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs_finetune"],
        callbacks=callbacks,
        verbose=2,
    )
    print(f"Finished Perch fine-tuning in {format_duration(perf_counter() - start)}")
    return history


def predict_dataset(model: tf.keras.Model, ds) -> tuple[np.ndarray, np.ndarray]:
    targets = []
    logits = []
    for x_batch, y_batch in ds:
        targets.append(y_batch.numpy())
        logits.append(model.predict(x_batch, verbose=0))
    return np.concatenate(targets, axis=0), np.concatenate(logits, axis=0)
