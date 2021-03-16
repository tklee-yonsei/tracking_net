from typing import Tuple

import tensorflow as tf
from models.gpu_check import check_first_gpu
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    Input,
    Layer,
    MaxPooling2D,
    ReLU,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model

check_first_gpu()


def encoder_conv(feature_map, filters=int):
    conv: Layer = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(
        feature_map
    )
    conv = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(conv)
    return conv


def decoder_upsample(feature_map, filters=int):
    up_sample = UpSampling2D(size=(2, 2))(feature_map)
    conv = Conv2D(
        filters, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(up_sample)
    return conv


def decoder_conv(feature_map, filters=int):
    conv: Layer = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(
        feature_map
    )
    conv = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(conv)
    return conv


def unet_l4_detach_activation(
    input_name: str = "unet_input",
    output_name: str = "unet_output",
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    alpha=1.0,
):
    filters: int = 16
    inputs = Input(shape=input_shape, name=input_name)

    # Encoder
    # 256 -> 128, 1 -> 64
    conv1 = encoder_conv(feature_map=inputs, filters=filters * 4)
    conv1_activate = ReLU()(conv1)
    pool1: Layer = MaxPooling2D(pool_size=(2, 2))(conv1_activate)

    # 128 -> 64, 64 -> 128
    conv2 = encoder_conv(feature_map=pool1, filters=filters * 8)
    conv2_activate = ReLU()(conv2)
    pool2: Layer = MaxPooling2D(pool_size=(2, 2))(conv2_activate)

    # 64 -> 32, 128 -> 256
    conv3 = encoder_conv(feature_map=pool2, filters=filters * 16)
    conv3_activate = ReLU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_activate)

    # Bottle Neck
    # 32 -> 32, 256 -> 512
    conv4 = encoder_conv(feature_map=pool3, filters=filters * 32)
    conv4_activate = ReLU()(conv4)
    drop4 = Dropout(0.5)(conv4_activate)

    # Decoder
    # 32 -> 64, 512 -> 256
    up1 = decoder_upsample(drop4, filters=filters * 16)
    merge1 = concatenate([conv3_activate, up1], axis=3)
    conv5 = decoder_conv(merge1, filters=filters * 16)
    conv5 = ReLU()(conv5)

    # 64 -> 128, 256 -> 128
    up2 = decoder_upsample(conv5, filters=filters * 8)
    merge2 = concatenate([conv2_activate, up2], axis=3)
    conv6 = decoder_conv(merge2, filters=filters * 8)
    conv6 = ReLU()(conv6)

    # 128 -> 256, 128 -> 64
    up3 = decoder_upsample(conv6, filters=filters * 4)
    merge3 = concatenate([conv1_activate, up3], axis=3)
    conv7 = decoder_conv(merge3, filters=filters * 4)
    conv7 = ReLU()(conv7)

    # 256 -> 256, 64 -> 2
    conv8 = Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv7)

    # 256 -> 256, 2 -> 1
    conv9 = Conv2D(1, 1, name=output_name, activation="sigmoid")(conv8)

    return Model(inputs=[inputs], outputs=[conv9])
