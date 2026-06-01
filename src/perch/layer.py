from __future__ import annotations

from pathlib import Path

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="birdclef")
class PerchEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        model_path: str | Path,
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__(trainable=trainable, **kwargs)
        self.model_path = str(model_path)
        self.saved_model_layer = tf.keras.layers.TFSMLayer(
            str(model_path),
            call_endpoint="serving_default",
            trainable=trainable,
        )

    def call(self, inputs, training=False):
        outputs = self.saved_model_layer(inputs, training=training)
        return outputs["embedding"]

    def get_config(self):
        config = super().get_config()
        config.update({"model_path": self.model_path})
        return config
