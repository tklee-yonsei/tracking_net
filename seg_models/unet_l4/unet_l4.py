from typing import Tuple

from keras import losses, optimizers
from keras.layers import *
from keras.models import *
from keras.optimizers import *


def unet_l4(
    pre_trained_weight_path: str = None,
    filters: int = 16,
    input_size: Tuple[int, int, int] = (256, 256, 1),
) -> Model:
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
    conv2: Layer = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    # 128 -> 128, 128 -> 128
    conv2: Layer = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv2)
    # 128 -> 64, 128 -> 128
    pool2: Layer = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 64 -> 64, 128 -> 256
    conv3: Layer = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)
    # 64 -> 64, 256 -> 256
    conv3 = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3)
    # 64 -> 32, 256 -> 256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 32 -> 32, 256 -> 512
    conv4 = Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool3)
    # 32 -> 32, 512 -> 512
    conv4 = Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4)
    # 32 -> 32, 512 -> 512
    drop4 = Dropout(0.5)(conv4)

    # 32 -> 64, 512 -> 256
    up7 = Conv2D(
        filters * 16,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(drop4))
    # (64, 64) -> 64, (256, 256) -> 512
    merge7 = concatenate([conv3, up7], axis=3)
    # 64 -> 64, 512 -> 256
    conv7 = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge7)
    # 64 -> 64, 256 -> 256
    conv7 = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv7)

    # 64 -> 128, 256 -> 128
    up8 = Conv2D(
        filters * 8,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(conv7))
    # (128, 128) -> 128, (128, 128) -> 256
    merge8 = concatenate([conv2, up8], axis=3)
    # 128 -> 128, 256 -> 128
    conv8 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge8)
    # 128 -> 128, 128 -> 128
    conv8 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv8)

    # 128 -> 256, 128 -> 64
    up9 = Conv2D(
        filters * 4,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(conv8))
    # (256, 256) -> 256, (64, 64) -> 128
    merge9 = concatenate([conv1, up9], axis=3)
    # 256 -> 256, 128 -> 64
    conv9 = Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge9)
    # 256 -> 256, 64 -> 64
    conv9 = Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv9)

    # 256 -> 256, 64 -> 2
    conv9 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    # 256 -> 256, 2 -> 1
    conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss=losses.binary_crossentropy,
        metrics=["accuracy"],
    )

    if pre_trained_weight_path:
        model.load_weights(pre_trained_weight_path)

    return model
