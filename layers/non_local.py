from typing import Optional

from keras import backend as K
from keras.layers import (
    Activation,
    Conv1D,
    Conv2D,
    Conv3D,
    Lambda,
    Layer,
    MaxPool1D,
    Reshape,
    add,
    dot,
)


def _convND(inputs, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(
            channels, 1, padding="same", use_bias=False, kernel_initializer="he_normal"
        )(inputs)
    elif rank == 4:
        x = Conv2D(
            channels,
            (1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(inputs)
    else:
        x = Conv3D(
            channels,
            (1, 1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(inputs)
    return x


def _reshape(inputs, input_shape, channel_dim, intermediate_dim):
    rank = len(input_shape)
    if rank == 3:
        _, dim1, _ = input_shape
        y = Reshape((dim1, intermediate_dim))(inputs)
    elif rank == 4:
        if channel_dim == -1:
            _, dim1, dim2, _ = input_shape
            y = Reshape((dim1, dim2, intermediate_dim))(inputs)
        else:
            _, _, dim1, dim2 = input_shape
            y = Reshape((intermediate_dim, dim1, dim2))(inputs)
    else:
        if channel_dim == -1:
            _, dim1, dim2, dim3, _ = input_shape
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(inputs)
        else:
            _, _, dim1, dim2, dim3 = input_shape
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(inputs)
    return y


class NonLocalBlock(Layer):
    def __init__(
        self,
        intermediate_dim: Optional[int] = None,
        mode: str = "embedded",
        sub_sample: bool = True,
        add_residual: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.mode = mode
        self.sub_sample = sub_sample
        self.add_residual = add_residual

    def build(self, batch_input_shape):
        self.input_shape = batch_input_shape
        self.channel_dim = 1 if K.image_data_format() == "channels_first" else -1
        super(NonLocalBlock, self).build(batch_input_shape)

    def call(self, inputs):
        batch_size = self.input_shape[0]
        channels = self.input_shape[-1]
        rank = len(self.input_shape)
        if self.intermediate_dim is None:
            self.intermediate_dim = channels // 2

        # 1) f path
        # Gaussian instantiation
        if self.mode == "gaussian":
            x1 = Reshape((-1, channels))(inputs)  # xi
            x2 = Reshape((-1, channels))(inputs)  # xj
            f = dot([x1, x2], axes=2)
            f = Activation("softmax")(f)

        # Dot instantiation
        elif self.mode == "dot":
            # theta path
            theta = _convND(inputs, rank, self.intermediate_dim)
            theta = Reshape((-1, self.intermediate_dim))(theta)

            # phi path
            phi = _convND(inputs, rank, self.intermediate_dim)
            phi = Reshape((-1, self.intermediate_dim))(phi)

            f = dot([theta, phi], axes=2)

            size = K.int_shape(f)

            # scale the values to make it size invariant
            f = Lambda(lambda z: (1.0 / float(size[-1])) * z)(f)

        # Concatenation instantiation
        elif self.mode == "concatenate":
            raise NotImplementedError("Concatenate model has not been implemented yet")

        # Embedded Gaussian instantiation
        else:
            # theta path
            theta = _convND(inputs, rank, self.intermediate_dim)
            theta = Reshape((-1, self.intermediate_dim))(theta)

            # phi path
            phi = _convND(inputs, rank, self.intermediate_dim)
            phi = Reshape((-1, self.intermediate_dim))(phi)

            if self.sub_sample:
                phi = MaxPool1D()(phi)

            f = dot([theta, phi], axes=2)
            f = Activation("softmax")(f)

        # 2) g path
        g = _convND(inputs, rank, self.intermediate_dim)
        g = Reshape((-1, self.intermediate_dim))(g)

        if self.sub_sample and self.mode == "embedded":
            # shielded computation
            g = MaxPool1D()(g)

        # 3) compute output path
        y = dot([f, g], axes=[2, 1])

        # reshape to input tensor format
        y = _reshape(y, self.input_shape, self.channel_dim, self.intermediate_dim)

        # project filters
        y = _convND(y, rank, channels)

        # residual connection
        if self.add_residual:
            y = add([inputs, y])

        return y
