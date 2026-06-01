from __future__ import annotations

import tensorflow as tf

from src.perch.layer import PerchEmbeddingLayer


def build_model(cfg: dict) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(160000,), dtype=tf.float32, name="waveform")
    embeddings = PerchEmbeddingLayer(
        cfg["perch_model_path"],
        trainable=False,
        name="perch",
    )(inputs)

    head_type = cfg.get("head_type", "mlp")
    x = tf.keras.layers.Dropout(cfg["dropout"], name="head_dropout_1")(embeddings)
    if head_type == "mlp":
        x = tf.keras.layers.Dense(cfg["hidden_dim"], activation="gelu", name="head_dense")(x)
        x = tf.keras.layers.Dropout(cfg["dropout"], name="head_dropout_2")(x)
    elif head_type != "linear":
        raise ValueError(f"Unknown head_type: {head_type}")
    logits = tf.keras.layers.Dense(234, name="birdclef_logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="perch_birdclef")


def build_embedding_model(cfg: dict) -> tf.keras.Model:
    embeddings = tf.keras.Input(shape=(1536,), dtype=tf.float32, name="perch_embedding")
    perch_scores = tf.keras.Input(shape=(234,), dtype=tf.float32, name="mapped_perch_scores")

    head_type = cfg.get("head_type", "mlp")
    x = tf.keras.layers.Dropout(cfg["dropout"], name="head_dropout_1")(embeddings)
    if head_type == "mlp":
        x = tf.keras.layers.Dense(cfg["hidden_dim"], activation="gelu", name="head_dense")(x)
        x = tf.keras.layers.Dropout(cfg["dropout"], name="head_dropout_2")(x)
    elif head_type != "linear":
        raise ValueError(f"Unknown head_type: {head_type}")
    logits = tf.keras.layers.Dense(234, name="birdclef_logits")(x)
    zero_perch_scores = tf.keras.layers.Lambda(lambda value: value * 0.0, name="ignore_mapped_perch_scores")(perch_scores)
    logits = tf.keras.layers.Add(name="birdclef_logits_with_cached_input")([logits, zero_perch_scores])
    return tf.keras.Model(inputs=[embeddings, perch_scores], outputs=logits, name="cached_perch_birdclef")
