from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf

from .metrics import challenge_score_from_arrays, sigmoid


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def weighted_binary_crossentropy(pos_weights: np.ndarray):
    weights = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, logits):
        per_class_loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=logits,
            pos_weight=weights,
        )
        return tf.reduce_mean(per_class_loss)

    return loss


class EpochTimingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = perf_counter() - self.epoch_start
        print(f" - epoch_time: {format_duration(elapsed)}")


class ChallengeScoreCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        val_ds,
        labels: list[str],
        row_indices: np.ndarray | None = None,
        num_rows: int | None = None,
        aggregation: str = "mean",
        top_k: int = 2,
    ):
        super().__init__()
        self.val_ds = val_ds
        self.labels = labels
        self.row_indices = row_indices
        self.num_rows = num_rows
        self.aggregation = aggregation
        self.top_k = top_k
        self.best = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.row_indices is None:
            y_true, logits = predict_dataset(self.model, self.val_ds)
            scores = sigmoid(logits)
        else:
            y_true, scores = predict_multicrop_dataset(
                self.model,
                self.val_ds,
                self.row_indices,
                self.num_rows,
                self.aggregation,
                self.top_k,
            )
        value = challenge_score_from_arrays(y_true, scores, self.labels)
        logs["val_challenge_score"] = value
        self.best = max(self.best, value)
        print(f" - val_challenge_score: {value:.5f}")


def compile_model(model: tf.keras.Model, learning_rate: float, cfg: dict, pos_weights: np.ndarray | None = None) -> None:
    loss_name = cfg["loss"]
    if loss_name == "bce":
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    elif loss_name == "weighted_bce":
        if pos_weights is None:
            raise ValueError("weighted_bce requires positive class weights")
        loss = weighted_binary_crossentropy(pos_weights)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
    )


def train_head_only(
    model,
    train_ds,
    val_ds,
    labels,
    cfg,
    pos_weights=None,
    val_row_indices=None,
    num_val_rows=None,
):
    print("Starting head-only training")
    start = perf_counter()
    compile_model(model, cfg["head_lr"], cfg, pos_weights)
    checkpoint_dir = Path("outputs") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EpochTimingCallback(),
        ChallengeScoreCallback(
            val_ds,
            labels,
            row_indices=val_row_indices,
            num_rows=num_val_rows,
            aggregation=cfg["validation_crop_aggregation"],
            top_k=cfg["validation_top_k"],
        ),
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


def predict_dataset(model: tf.keras.Model, ds) -> tuple[np.ndarray, np.ndarray]:
    targets = []
    logits = []
    for x_batch, y_batch in ds:
        targets.append(y_batch.numpy())
        logits.append(model.predict(x_batch, verbose=0))
    return np.concatenate(targets, axis=0), np.concatenate(logits, axis=0)


def aggregate_crop_scores(
    scores: np.ndarray,
    row_indices: np.ndarray,
    num_rows: int,
    aggregation: str,
    top_k: int,
) -> np.ndarray:
    output = np.zeros((num_rows, scores.shape[1]), dtype=np.float32)
    for row_idx in range(num_rows):
        row_scores = scores[row_indices == row_idx]
        if aggregation == "mean":
            output[row_idx] = row_scores.mean(axis=0)
        elif aggregation == "max":
            output[row_idx] = row_scores.max(axis=0)
        elif aggregation == "top_k_mean":
            k = min(top_k, len(row_scores))
            output[row_idx] = np.sort(row_scores, axis=0)[-k:].mean(axis=0)
        else:
            raise ValueError(f"Unknown validation_crop_aggregation: {aggregation}")
    return output


def predict_multicrop_dataset(
    model: tf.keras.Model,
    ds,
    row_indices: np.ndarray,
    num_rows: int,
    aggregation: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    crop_targets, crop_logits = predict_dataset(model, ds)
    crop_scores = sigmoid(crop_logits)
    scores = aggregate_crop_scores(crop_scores, row_indices, num_rows, aggregation, top_k)
    targets = np.zeros((num_rows, crop_targets.shape[1]), dtype=np.float32)
    for row_idx in range(num_rows):
        targets[row_idx] = crop_targets[np.flatnonzero(row_indices == row_idx)[0]]
    return targets, scores
