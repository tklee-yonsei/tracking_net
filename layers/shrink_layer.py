import tensorflow as tf
from tensorflow.keras.layers import Layer
from utils.tf_images import tf_image_shrink


class ShrinkLayer(Layer):
    def __init__(
        self, bin_num: int, resize_by_power_of_two: int, **kwargs,
    ):
        super().__init__(**kwargs)
        self.bin_num: int = bin_num
        self.resize_by_power_of_two: int = resize_by_power_of_two

    def build(self, input_shape):
        ratio = 2 ** self.resize_by_power_of_two
        self.img_wh = input_shape[1] // ratio

    def get_config(self):
        config = super(ShrinkLayer, self).get_config()
        config.update(
            {
                "bin_num": self.bin_num,
                "resize_by_power_of_two": self.resize_by_power_of_two,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        r = tf_image_shrink(inputs, self.bin_num, self.resize_by_power_of_two)
        r = tf.reshape(r, (-1, self.img_wh, self.img_wh, self.bin_num))
        return r
