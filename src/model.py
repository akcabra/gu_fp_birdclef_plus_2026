from __future__ import annotations

import tensorflow as tf

from .perch import PerchEmbeddingLayer


def build_model(cfg: dict, trainable_backbone: bool = False) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(160000,), dtype=tf.float32, name="waveform")
    embeddings = PerchEmbeddingLayer(
        cfg["perch_model_path"],
        trainable=trainable_backbone,
        name="perch",
    )(inputs)

    x = tf.keras.layers.Dropout(cfg["dropout"], name="head_dropout_1")(embeddings)
    if cfg.get("head_type", "mlp") == "mlp":
        x = tf.keras.layers.Dense(cfg["hidden_dim"], activation="gelu", name="head_dense")(x)
        x = tf.keras.layers.Dropout(cfg["dropout"], name="head_dropout_2")(x)
    logits = tf.keras.layers.Dense(234, name="birdclef_logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="perch_birdclef")


def set_backbone_trainable(model: tf.keras.Model, trainable: bool) -> None:
    perch = model.get_layer("perch")
    perch.trainable = trainable
