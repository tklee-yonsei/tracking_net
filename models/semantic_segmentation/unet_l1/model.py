from typing import Tuple

from keras import losses, optimizers
from keras.layers import (
    Conv2D,
    Dropout,
    Input,
    Layer,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)
from keras.models import Model

from models.gpu_check import check_first_gpu

check_first_gpu()


def unet_l1(
    input_name: str, input_shape: Tuple[int, int, int], output_name: str, alpha=1.0
):
    filters: int = 16
    inputs = Input(shape=input_shape, name=input_name)

    # 256 -> 256, 1 -> 64
    conv1: Layer = Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    # 256 -> 256, 64 -> 64
    conv1: Layer = Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)
    # 256 -> 128, 64 -> 64
    pool1: Layer = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 128 -> 128, 64 -> 128
    conv2 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    # 128 -> 128, 128 -> 128
    conv2 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv2)
    # 128 -> 128, 128 -> 128
    drop2 = Dropout(0.5)(conv2)

    # 128 -> 256, 128 -> 64
    up3 = Conv2D(
        filters * 4,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(drop2))
    # (256, 256) -> 256, (64, 64) -> 128
    merge3 = concatenate([conv1, up3], axis=3)
    # 256 -> 256, 128 -> 64
    conv4 = Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge3)
    # 256 -> 256, 64 -> 64
    conv4 = Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4)

    # 256 -> 256, 64 -> 2
    conv5 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv4)
    # 256 -> 256, 2 -> 1
    conv6 = Conv2D(1, 1, name=output_name, activation="sigmoid")(conv5)

    return Model(inputs=[inputs], outputs=[conv6])
