from typing import Tuple

from keras import losses, optimizers
from keras.layers import (
    Conv2D,
    Cropping2D,
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


def vanilla_unet(
    input_name: str, input_shape: Tuple[int, int, int], output_name: str, alpha=1.0
) -> Model:
    inputs = Input(shape=input_shape, name=input_name)

    # 572 -> 570, 1 -> 64
    conv1 = Conv2D(64, 3, activation="relu", kernel_initializer="he_normal")(inputs)
    # 570 -> 568, 64 -> 64
    conv1 = Conv2D(64, 3, activation="relu", kernel_initializer="he_normal")(conv1)

    # 568 -> 284, 64 -> 64
    pool1 = MaxPooling2D()(conv1)
    # 284 -> 282, 64 -> 128
    conv2 = Conv2D(128, 3, activation="relu", kernel_initializer="he_normal")(pool1)
    # 282 -> 280, 128 -> 128
    conv2 = Conv2D(128, 3, activation="relu", kernel_initializer="he_normal")(conv2)

    # 280 -> 140, 128 -> 128
    pool2 = MaxPooling2D()(conv2)
    # 140 -> 138, 128 -> 256
    conv3 = Conv2D(256, 3, activation="relu", kernel_initializer="he_normal")(pool2)
    # 138 -> 136, 256 -> 256
    conv3 = Conv2D(256, 3, activation="relu", kernel_initializer="he_normal")(conv3)

    # 136 -> 68, 256 -> 256
    pool3 = MaxPooling2D()(conv3)
    # 68 -> 66, 256 -> 512
    conv4 = Conv2D(512, 3, activation="relu", kernel_initializer="he_normal")(pool3)
    # 66 -> 64, 512 -> 512
    conv4 = Conv2D(512, 3, activation="relu", kernel_initializer="he_normal")(conv4)

    # 64 -> 32, 512 -> 512
    pool4 = MaxPooling2D()(conv4)
    # 32 -> 30, 512 -> 1024
    conv5 = Conv2D(1024, 3, activation="relu", kernel_initializer="he_normal")(pool4)
    # 30 -> 28, 1024 -> 1024
    conv5 = Conv2D(1024, 3, activation="relu", kernel_initializer="he_normal")(conv5)

    # 28 -> 56, 1024 -> 512
    up6 = Conv2D(
        512, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D()(conv5))
    # 64 -> 56, 512 -> 512
    conv4 = Cropping2D(cropping=(4, 4))(conv4)
    # (56, 56) -> 56, (512, 512) -> 1024
    merge6 = concatenate([conv4, up6], axis=3)
    # 56 -> 54, 1024 -> 512
    conv6 = Conv2D(512, 3, activation="relu", kernel_initializer="he_normal")(merge6)
    # 54 -> 52, 512 -> 512
    conv6 = Conv2D(512, 3, activation="relu", kernel_initializer="he_normal")(conv6)

    # 52 -> 104, 512 -> 256
    up7 = Conv2D(
        256, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D()(conv6))
    # 136 -> 104, 256 -> 256
    conv3 = Cropping2D(cropping=(16, 16))(conv3)
    # (104, 104) -> 104, (256, 256) -> 512
    merge7 = concatenate([conv3, up7], axis=3)
    # 104 -> 102, 512 -> 256
    conv7 = Conv2D(256, 3, activation="relu", kernel_initializer="he_normal")(merge7)
    # 102 -> 100, 256 -> 256
    conv7 = Conv2D(256, 3, activation="relu", kernel_initializer="he_normal")(conv7)

    # 100 -> 200, 256 -> 128
    up8 = Conv2D(
        128, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D()(conv7))
    # 280 -> 200, 128 -> 128
    conv2 = Cropping2D(cropping=(40, 40))(conv2)
    # (200, 200) -> 200, (128, 128) -> 256
    merge8 = concatenate([conv2, up8], axis=3)
    # 200 -> 198, 256 -> 128
    conv8 = Conv2D(128, 3, activation="relu", kernel_initializer="he_normal")(merge8)
    # 198 -> 196, 128 -> 128
    conv8 = Conv2D(128, 3, activation="relu", kernel_initializer="he_normal")(conv8)

    # 196 -> 392, 128 -> 64
    up9 = Conv2D(
        64, 2, activation="relu", padding="same", kernel_initializer="he_normal"
    )(UpSampling2D()(conv8))
    # 568 -> 392, 64 -> 64
    conv1 = Cropping2D(cropping=(88, 88))(conv1)
    # (392, 392) -> 392, (64, 64) -> 128
    merge9 = concatenate([conv1, up9], axis=3)
    # 392 -> 390, 128 -> 64
    conv9 = Conv2D(64, 3, activation="relu", kernel_initializer="he_normal")(merge9)
    # 390 -> 388, 64 -> 64
    conv9 = Conv2D(64, 3, activation="relu", kernel_initializer="he_normal")(conv9)
    # 388 -> 388, 64 -> 2
    conv9 = Conv2D(2, 1, activation="relu", kernel_initializer="he_normal")(conv9)

    # 388 -> 386, 2 -> 1
    outputs = Conv2D(1, 1, name=output_name, activation="sigmoid")(conv9)

    return Model(inputs=[inputs], outputs=[outputs])
