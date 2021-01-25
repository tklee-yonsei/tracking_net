from typing import Optional

import tensorflow as tf
from image_keras.tf.keras.layers.extract_patch_layer import ExtractPatchLayer
from tensorflow.keras.layers import Conv2D, Layer, Reshape, Softmax


class RefLocal6(Layer):
    """
    Deprecated.

    """

    def __init__(
        self,
        bin_size: int,
        intermediate_dim: Optional[int] = None,
        k_size: int = 5,
        mode: str = "dot",
        aggregate_mode: str = "weighted_sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.input_intermediate_dim = intermediate_dim
        self.k_size = k_size
        self.mode = mode
        self.aggregate_mode = aggregate_mode

    def build(self, input_shape):
        self.input_main_batch_size = input_shape[0][0]
        self.input_main_h_size = input_shape[0][1]
        self.input_main_channels_size = input_shape[0][-1]
        self.input_intermediate_dim = (
            self.input_intermediate_dim or self.input_main_channels_size
        )

        self.conv_main = Conv2D(
            filters=self.input_intermediate_dim,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.conv_ref = Conv2D(
            filters=self.input_intermediate_dim,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )

    def get_config(self):
        config = super(RefLocal6, self).get_config()
        config.update(
            {
                "bin_size": self.bin_size,
                "intermediate_dim": self.input_intermediate_dim,
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
        ref = inputs[1]
        ref_value = inputs[2]

        conv_main = self.conv_main(main)
        conv_ref = self.conv_ref(ref)

        # Attention
        ref_stacked = ExtractPatchLayer(k_size=self.k_size)(conv_ref)
        ref_stacked = tf.reshape(
            ref_stacked,
            (
                -1,
                self.input_main_h_size,
                self.input_main_h_size,
                self.k_size * self.k_size,
                self.input_main_channels_size,
            ),
        )
        if self.mode == "dot":
            attn = tf.einsum("bhwc,bhwkc->bhwk", conv_main, ref_stacked)
            attn = tf.nn.softmax(attn, axis=-1)
        elif self.mode == "norm_dot":
            ref_main = tf.concat([ref_stacked, tf.expand_dims(main, -2)], axis=-2)
            attn = tf.einsum("bhwc,bhwkc->bhwk", main, ref_main)
            attn = tf.nn.softmax(attn, axis=-1)
            attn = attn[:, :, :, :-1]
        elif self.mode == "gaussian":
            attn = tf.math.exp(
                -tf.reduce_sum(tf.math.squared_difference(main, ref_stacked), axis=-1)
            )
            raise NotImplementedError("gaussian model has not been implemented yet")
        else:
            raise ValueError(
                "`mode` value is not valid. Should be one of 'dot', 'norm_dot', 'gaussian'."
            )

        # Aggregate
        if self.aggregate_mode == "weighted_sum":
            ref_value_stacked = ExtractPatchLayer(k_size=self.k_size)(ref_value)
            ref_value_stacked = tf.reshape(
                ref_value_stacked,
                (
                    -1,
                    self.input_main_h_size,
                    self.input_main_h_size,
                    self.k_size * self.k_size,
                    self.bin_size,
                ),
            )
            aggregation = tf.einsum("bhwk,bhwki->bhwi", attn, ref_value_stacked)
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
            attn2 = tf.one_hot(
                tf.argmax(attn, axis=-1), depth=tf.shape(attn)[-1], dtype=tf.float32
            )
            one_hot = tf.one_hot(tf.cast(output, tf.int32), num_words, dtype=tf.float32)
            aggregation = tf.einsum("bhwk,bhwki->bhwi", attn2, ref_value_stacked)
        else:
            raise ValueError(
                "`aggregate_mode` value is not valid. Should be one of 'weighted_sum', 'argmax'."
            )
        return aggregation
