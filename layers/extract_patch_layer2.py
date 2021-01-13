from tensorflow.keras.layers import Layer
from utils.tf_images import tf_extract_patches


class ExtractPatchLayer2(Layer):
    def __init__(self, k_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        if (k_size % 2) == 0:
            raise RuntimeError("`k_size` should be odd number.")
        self.k_size = k_size

    def build(self, input_shape):
        self.img_wh = input_shape[1]
        self.channel = input_shape[-1]

    def get_config(self):
        config = super(ExtractPatchLayer2, self).get_config()
        config.update({"k_size": self.k_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        return tf_extract_patches(
            inputs, ksize=self.k_size, img_wh=self.img_wh, channel=self.channel
        )
