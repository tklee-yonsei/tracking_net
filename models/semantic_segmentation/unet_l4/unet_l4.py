from typing import List, Tuple

import keras
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
from keras.metrics import Metric
from keras.models import Model
from keras.optimizers import Adam, Optimizer

from idl.custom.metrics import BinaryClassMeanIoU
from idl.model_manager import LossDescriptor, ModelDescriptor, ModelHelper

unet_l4_model_descriptor_default: ModelDescriptor = ModelDescriptor(
    inputs=[("input", (256, 256, 1))], outputs=[("output", (256, 256, 1))]
)

unet_l4_loss_descriptors_default: List[LossDescriptor] = [
    LossDescriptor(loss=keras.losses.BinaryCrossentropy(), weight=1.0)
]


class UnetL4ModelHelper(ModelHelper):
    def __init__(
        self,
        model_descriptor: ModelDescriptor = unet_l4_model_descriptor_default,
        alpha: float = 1.0,
    ):
        super().__init__(model_descriptor, alpha)

    def get_model(self) -> Model:
        return unet_l4(
            input_name=self.model_descriptor.inputs[0][0],
            input_shape=self.model_descriptor.inputs[0][1],
            output_name=self.model_descriptor.outputs[0][0],
            alpha=self.alpha,
        )

    def compile_model(
        self,
        model: Model,
        optimizer: Optimizer = Adam(lr=1e-4),
        loss_list: List[LossDescriptor] = unet_l4_loss_descriptors_default,
        metrics: List[Metric] = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            BinaryClassMeanIoU(name="mean_iou"),
        ],
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        **kwargs
    ) -> Model:
        return super().compile_model(
            model=model,
            optimizer=optimizer,
            loss_list=loss_list,
            metrics=metrics,
            model_descriptor=self.model_descriptor,
            sample_weight_mode=sample_weight_mode,
            weighted_metrics=weighted_metrics,
            target_tensors=target_tensors,
            **kwargs
        )


def unet_l4(
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
    conv10 = Conv2D(1, 1, name=output_name, activation="sigmoid")(conv9)

    return Model(inputs=[inputs], outputs=[conv10])
