from typing import Tuple

from keras import losses, optimizers
from keras.layers import *
from keras.models import *
from keras.optimizers import *


def unet_l1() -> Model:
    input_size: Tuple[int, int, int] = (256, 256, 1)
    filters: int = 16

    inputs = Input(input_size)

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
    conv6 = Conv2D(1, 1, activation="sigmoid")(conv5)

    return Model(inputs=inputs, outputs=conv6)
