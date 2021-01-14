from typing import Optional

import tensorflow as tf
from image_keras.tf.keras.layers.extract_patch_layer import ExtractPatchLayer
from tensorflow.keras.layers import Conv2D, Layer


class RefLocal4(Layer):
    def __init__(
        self,
        bin_size: int,
        k_size: int = 5,
        mode: str = "dot",
        aggregate_mode: str = "weighted_sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.k_size = k_size
        self.mode = mode
        self.aggregate_mode = aggregate_mode

    def build(self, input_shape):
        self.custom_input_shape = input_shape
        self.custom_shape = input_shape[0]
        self.batch_size = input_shape[0][0]
        self.h_size = input_shape[0][-3]
        self.channels_size = input_shape[0][-1]

    def get_config(self):
        config = super(RefLocal4, self).get_config()
        config.update(
            {
                "bin_size": self.bin_size,
                "k_size": self.k_size,
                "mode": self.mode,
                "aggregate_mode": self.aggregate_mode,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        main = inputs[0]
        main_value = inputs[1]
        ref = inputs[2]
        ref_value = inputs[3]

        # Attention
        ref_stacked = ExtractPatchLayer(k_size=self.k_size)(ref)
        ref_stacked = tf.reshape(
            ref_stacked,
            (
                -1,
                self.h_size,
                self.h_size,
                self.k_size * self.k_size,
                self.channels_size,
            ),
        )
        if self.mode == "dot":
            ref_main = tf.concat([ref_stacked, tf.expand_dims(main, -2)], axis=-2)
            attn = tf.einsum("bhwc,bhwkc->bhwk", main, ref_main)
            attn = tf.nn.softmax(attn, axis=-1)
        elif self.mode == "gaussian":
            attn = tf.math.exp(
                -tf.reduce_sum(tf.math.squared_difference(main, ref_stacked), axis=-1)
            )
            raise NotImplementedError("gaussian model has not been implemented yet")
        else:
            raise ValueError(
                "`mode` value is not valid. Should be one of 'dot', 'gaussian'."
            )

        # Aggregate
        if self.aggregate_mode == "weighted_sum":
            ref_value_stacked = ExtractPatchLayer(k_size=self.k_size)(ref_value)
            ref_value_stacked = tf.reshape(
                ref_value_stacked,
                (
                    -1,
                    self.h_size,
                    self.h_size,
                    self.k_size * self.k_size,
                    self.bin_size,
                ),
            )
            ref_main_value = tf.concat(
                [ref_value_stacked, tf.expand_dims(main_value, -2)], axis=-2
            )
            aggregation = tf.einsum("bhwk,bhwki->bhwi", attn, ref_main_value)
        elif self.aggregate_mode == "argmax":
            raise NotImplementedError("`argmax` aggregate has not been implemented yet")
            ref_value_stacked = ExtractPatchLayer(k_size=self.k_size)(ref_value)
            ref_value_stacked = tf.reshape(
                ref_value_stacked,
                (
                    -1,
                    self.h_size,
                    self.h_size,
                    self.k_size * self.k_size,
                    self.bin_size,
                ),
            )
            ref_main_value = tf.concat(
                [ref_value_stacked, tf.expand_dims(main_value, -2)], axis=-2
            )
            attn2 = tf.one_hot(
                tf.argmax(attn, axis=-1), depth=tf.shape(attn)[-1], dtype=tf.float32
            )
            aggregation = tf.einsum("bhwk,bhwki->bhwi", attn2, ref_main_value)
        else:
            raise ValueError(
                "`aggregate_mode` value is not valid. Should be one of 'weighted_sum', 'argmax'."
            )
        return aggregation
