from __future__ import annotations

from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub


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
        self.hub_layer = hub.KerasLayer(
            str(model_path),
            output_key="embedding",
            trainable=trainable,
        )

    def call(self, inputs, training=False):
        return self.hub_layer(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"model_path": self.model_path})
        return config
