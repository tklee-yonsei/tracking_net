from typing import Optional

import tensorflow as tf
from image_keras.tf.keras.layers.extract_patch_layer import ExtractPatchLayer
from tensorflow.keras.layers import Conv2D, Layer, Reshape, Softmax


class RefLocal5(Layer):
    """
    Ref Local layer includes intermediate conv layer.
    With Non local like aggregation.
    """

    def __init__(
        self, intermediate_dim: int, k_size: int = 5, mode: str = "dot", **kwargs
    ):
        super().__init__(**kwargs)
        self.k_size = k_size
        self.intermediate_dim = intermediate_dim
        self.mode = mode

    def build(self, input_shape):
        self.custom_input_shape = input_shape
        self.custom_shape = input_shape[0]
        self.batch_size = input_shape[0][0]
        self.h_size = input_shape[0][1]
        self.channels_size = input_shape[0][-1]

        self.conv_main = Conv2D(
            filters=self.intermediate_dim,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.conv_ref = Conv2D(
            filters=self.intermediate_dim,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.g_conv = Conv2D(
            filters=self.channels_size,
            kernel_size=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )

    def get_config(self):
        config = super(RefLocal5, self).get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "k_size": self.k_size,
                "mode": self.mode,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        main = inputs[0]
        ref = inputs[1]

        conv_main = self.conv_main(main)
        conv_ref = self.conv_ref(ref)

        # 1) f path
        ref_stacked = ExtractPatchLayer(k_size=self.k_size)(conv_ref)
        ref_stacked = Reshape(
            (self.h_size, self.h_size, self.k_size * self.k_size, self.intermediate_dim)
        )(ref_stacked)
        if self.mode == "dot":
            attn = tf.einsum("bhwc,bhwkc->bhwk", conv_main, ref_stacked)
            attn = Softmax()(attn)
            # attn = Reshape(
            #     (-1, tf.shape(attn)[1], tf.shape(attn)[2], self.k_size, self.k_size)
            # )(attn)
        elif self.mode == "gaussian":
            attn = tf.math.exp(
                -tf.reduce_sum(tf.math.squared_difference(main, ref_stacked), axis=-1)
            )
            raise NotImplementedError("gaussian model has not been implemented yet")
        else:
            raise ValueError(
                "`mode` value is not valid. Should be one of 'dot', 'gaussian'."
            )

        # 2) g path
        g = self.g_conv(attn)
        # y = tf.einsum("bhwkk,bhwd->bhwd", attn, g)
        # y = tf.einsum("bhwkk,bkkd->bhwd", attn, g)
        # y = dot([f, g], axes=[2, 1])
        # g = Reshape((-1, self.intermediate_dim))(g)

        return g
