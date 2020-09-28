from typing import Optional

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Conv2D, Lambda, Layer, MaxPool1D, Reshape, add, dot


class RefNonLocal(Layer):
    def __init__(
        self,
        intermediate_dim: Optional[int] = None,
        mode: str = "embedded",
        sub_sample: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.mode = mode
        self.sub_sample = sub_sample

    def build(self, input_shape):
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        self.channels_size = self.input_shape[-1]
        super(RefNonLocal, self).build(input_shape)

    def call(self, inputs):
        # 1) f path
        # Gaussian
        if self.mode == "gaussian":
            x1 = Reshape((-1, self.channels_size))(inputs)  # xi
            x2 = Reshape((-1, self.channels_size))(inputs)  # xj
            f = dot([x1, x2], axes=2)
            f = Activation("softmax")(f)

        # Dot
        elif self.mode == "dot":
            # theta path
            theta = Conv2D(
                self.channels_size,
                (1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(inputs)
            theta = Reshape((-1, self.intermediate_dim))(theta)

            # phi path
            phi = Conv2D(
                self.channels_size,
                (1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(inputs)
            phi = Reshape((-1, self.intermediate_dim))(phi)

            f = dot([theta, phi], axes=2)

            size = K.int_shape(f)

            f = Lambda(lambda z: (1.0 / float(size[-1])) * z)(f)

        # Concatenate
        elif self.mode == "concatenate":
            raise NotImplementedError("concatenate model has not been implemented yet")

        # Embedded Gaussian instantiation
        else:
            # theta path
            theta = Conv2D(
                self.channels_size,
                (1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(inputs)
            theta = Reshape((-1, self.intermediate_dim))(theta)

            # phi path
            phi = Conv2D(
                self.channels_size,
                (1, 1),
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
            )(inputs)
            phi = Reshape((-1, self.intermediate_dim))(phi)

            if self.sub_sample:
                phi = MaxPool1D()(phi)

            f = dot([theta, phi], axes=2)
            f = Activation("softmax")(f)

        # 2) g path
        g = Conv2D(
            self.channels_size,
            (1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(inputs)
        g = Reshape((-1, self.intermediate_dim))(g)

        if self.sub_sample and self.mode == "embedded":
            g = MaxPool1D()(g)

        # 3) compute output path
        y = dot([f, g], axes=[2, 1])

        # reshape to input tensor format
        y = Reshape((self.input_shape[1], self.input_shape[2], self.intermediate_dim))(
            y
        )

        # project filters
        theta = Conv2D(
            self.channels_size,
            (1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(y)

        # residual connection
        y = add([inputs, y])

        return y

    def compute_output_shape(self, input_shape):
        return input_shape
