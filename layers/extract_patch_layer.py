import tensorflow as tf
from tensorflow.keras.layers import Layer


class ExtractPatchLayer(Layer):
    def __init__(self, k_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k_size = k_size

    def get_config(self):
        config = super(ExtractPatchLayer, self).get_config()
        config.update({"k_size": self.k_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        return tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.k_size, self.k_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
