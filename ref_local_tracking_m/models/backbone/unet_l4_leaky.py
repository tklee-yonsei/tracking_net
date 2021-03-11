from typing import Tuple

from models.gpu_check import check_first_gpu
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    Input,
    Layer,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    LeakyReLU,
)
from tensorflow.keras.models import Model

check_first_gpu()


def unet_l4_leaky(
    input_name: str = "unet_input",
    output_name: str = "unet_output",
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    alpha=1.0,
):
    filters: int = 16
    inputs = Input(shape=input_shape, name=input_name)

    # 256 -> 256, 1 -> 64
    conv1: Layer = Conv2D(
        filters * 4, 3, padding="same", kernel_initializer="he_normal",
    )(inputs)
    conv1 = LeakyReLU()(conv1)
    # 256 -> 256, 64 -> 64
    conv1 = Conv2D(filters * 4, 3, padding="same", kernel_initializer="he_normal",)(
        conv1
    )
    conv1 = LeakyReLU()(conv1)
    # 256 -> 128, 64 -> 64
    pool1: Layer = MaxPooling2D(pool_size=(2, 2))(conv1)

    # 128 -> 128, 64 -> 128
    conv2: Layer = Conv2D(
        filters * 8, 3, padding="same", kernel_initializer="he_normal",
    )(pool1)
    conv2 = LeakyReLU()(conv2)
    # 128 -> 128, 128 -> 128
    conv2 = Conv2D(filters * 8, 3, padding="same", kernel_initializer="he_normal",)(
        conv2
    )
    conv2 = LeakyReLU()(conv2)
    # 128 -> 64, 128 -> 128
    pool2: Layer = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 64 -> 64, 128 -> 256
    conv3: Layer = Conv2D(
        filters * 16, 3, padding="same", kernel_initializer="he_normal",
    )(pool2)
    conv3 = LeakyReLU()(conv3)
    # 64 -> 64, 256 -> 256
    conv3 = Conv2D(filters * 16, 3, padding="same", kernel_initializer="he_normal",)(
        conv3
    )
    conv3 = LeakyReLU()(conv3)
    # 64 -> 32, 256 -> 256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # 32 -> 32, 256 -> 512
    conv4 = Conv2D(filters * 32, 3, padding="same", kernel_initializer="he_normal",)(
        pool3
    )
    conv4 = LeakyReLU()(conv4)
    # 32 -> 32, 512 -> 512
    conv4 = Conv2D(filters * 32, 3, padding="same", kernel_initializer="he_normal",)(
        conv4
    )
    conv4 = LeakyReLU()(conv4)
    # 32 -> 32, 512 -> 512
    drop4 = Dropout(0.5)(conv4)

    # 32 -> 64, 512 -> 256
    up7 = UpSampling2D(size=(2, 2))(drop4)
    up7 = Conv2D(filters * 16, 2, padding="same", kernel_initializer="he_normal",)(up7)
    up7 = LeakyReLU()(up7)
    # (64, 64) -> 64, (256, 256) -> 512
    merge7 = concatenate([conv3, up7], axis=3)
    # 64 -> 64, 512 -> 256
    conv7 = Conv2D(filters * 16, 3, padding="same", kernel_initializer="he_normal",)(
        merge7
    )
    conv7 = LeakyReLU()(conv7)
    # 64 -> 64, 256 -> 256
    conv7 = Conv2D(filters * 16, 3, padding="same", kernel_initializer="he_normal",)(
        conv7
    )
    conv7 = LeakyReLU()(conv7)

    # 64 -> 128, 256 -> 128
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(filters * 8, 2, padding="same", kernel_initializer="he_normal",)(up8)
    up8 = LeakyReLU()(up8)
    # (128, 128) -> 128, (128, 128) -> 256
    merge8 = concatenate([conv2, up8], axis=3)
    # 128 -> 128, 256 -> 128
    conv8 = Conv2D(filters * 8, 3, padding="same", kernel_initializer="he_normal",)(
        merge8
    )
    conv8 = LeakyReLU()(conv8)
    # 128 -> 128, 128 -> 128
    conv8 = Conv2D(filters * 8, 3, padding="same", kernel_initializer="he_normal",)(
        conv8
    )
    conv8 = LeakyReLU()(conv8)

    # 128 -> 256, 128 -> 64
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(filters * 4, 2, padding="same", kernel_initializer="he_normal",)(up9)
    up9 = LeakyReLU()(up9)
    # (256, 256) -> 256, (64, 64) -> 128
    merge9 = concatenate([conv1, up9], axis=3)
    # 256 -> 256, 128 -> 64
    conv9 = Conv2D(filters * 4, 3, padding="same", kernel_initializer="he_normal",)(
        merge9
    )
    conv9 = LeakyReLU()(conv9)
    # 256 -> 256, 64 -> 64
    conv9 = Conv2D(filters * 4, 3, padding="same", kernel_initializer="he_normal",)(
        conv9
    )
    conv9 = LeakyReLU()(conv9)

    # 256 -> 256, 64 -> 2
    conv9 = Conv2D(2, 3, padding="same", kernel_initializer="he_normal")(conv9)
    conv9 = LeakyReLU()(conv9)
    # 256 -> 256, 2 -> 1
    conv10 = Conv2D(1, 1, name=output_name, activation="sigmoid")(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
