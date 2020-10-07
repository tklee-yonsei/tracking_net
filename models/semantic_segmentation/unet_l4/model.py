from typing import Tuple

# from keras import losses, optimizers
# from keras.layers import (
#     Conv2D,
#     Dropout,
#     Input,
#     Layer,
#     MaxPooling2D,
#     UpSampling2D,
#     concatenate,
# )
# from keras.models import Model
import tensorflow as tf

from models.gpu_check import check_first_gpu

check_first_gpu()


def unet_l4(
    input_name: str, input_shape: Tuple[int, int, int], output_name: str, alpha=1.0
):
    filters: int = 16
    inputs = tf.keras.layers.Input(shape=input_shape, name=input_name)

    # 256 -> 256, 1 -> 64
    conv1: tf.keras.layers.Layer = tf.keras.layers.Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    # 256 -> 256, 64 -> 64
    conv1: tf.keras.layers.Layer = tf.keras.layers.Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)
    # 256 -> 128, 64 -> 64
    pool1: tf.keras.layers.Layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # 128 -> 128, 64 -> 128
    conv2: tf.keras.layers.Layer = tf.keras.layers.Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    # 128 -> 128, 128 -> 128
    conv2: tf.keras.layers.Layer = tf.keras.layers.Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv2)
    # 128 -> 64, 128 -> 128
    pool2: tf.keras.layers.Layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # 64 -> 64, 128 -> 256
    conv3: tf.keras.layers.Layer = tf.keras.layers.Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)
    # 64 -> 64, 256 -> 256
    conv3 = tf.keras.layers.Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3)
    # 64 -> 32, 256 -> 256
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # 32 -> 32, 256 -> 512
    conv4 = tf.keras.layers.Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool3)
    # 32 -> 32, 512 -> 512
    conv4 = tf.keras.layers.Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4)
    # 32 -> 32, 512 -> 512
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)

    # 32 -> 64, 512 -> 256
    up7 = tf.keras.layers.Conv2D(
        filters * 16,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(tf.keras.layers.UpSampling2D(size=(2, 2))(drop4))
    # (64, 64) -> 64, (256, 256) -> 512
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    # 64 -> 64, 512 -> 256
    conv7 = tf.keras.layers.Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge7)
    # 64 -> 64, 256 -> 256
    conv7 = tf.keras.layers.Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv7)

    # 64 -> 128, 256 -> 128
    up8 = tf.keras.layers.Conv2D(
        filters * 8,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    # (128, 128) -> 128, (128, 128) -> 256
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    # 128 -> 128, 256 -> 128
    conv8 = tf.keras.layers.Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge8)
    # 128 -> 128, 128 -> 128
    conv8 = tf.keras.layers.Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv8)

    # 128 -> 256, 128 -> 64
    up9 = tf.keras.layers.Conv2D(
        filters * 4,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    # (256, 256) -> 256, (64, 64) -> 128
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    # 256 -> 256, 128 -> 64
    conv9 = tf.keras.layers.Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge9)
    # 256 -> 256, 64 -> 64
    conv9 = tf.keras.layers.Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv9)

    # 256 -> 256, 64 -> 2
    conv9 = tf.keras.layers.Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    # 256 -> 256, 2 -> 1
    conv10 = tf.keras.layers.Conv2D(1, 1, name=output_name, activation="sigmoid")(conv9)

    return tf.keras.models.Model(inputs=[inputs], outputs=[conv10])
