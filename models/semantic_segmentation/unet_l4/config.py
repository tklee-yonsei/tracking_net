from typing import Callable, List

import keras
import tensorflow as tf
import numpy as np
from image_keras.custom.metrics import BinaryClassMeanIoU
from image_keras.model_manager import LossDescriptor, ModelDescriptor, ModelHelper
from image_keras.utils.image_transform import gray_image_apply_clahe, img_to_minmax

# from keras.metrics import Metric
# from keras.models import Model
# from keras.optimizers import Adam, Optimizer

from models.semantic_segmentation.unet_l4.model import unet_l4

unet_l4_model_descriptor_default: ModelDescriptor = ModelDescriptor(
    inputs=[("input", (256, 256, 1))], outputs=[("output", (256, 256, 1))]
)

unet_l4_loss_descriptors_default: List[LossDescriptor] = [
    LossDescriptor(loss=keras.losses.BinaryCrossentropy(), weight=1.0)
]

input_image_preprocessing_function: Callable[
    [np.ndarray], np.ndarray
] = gray_image_apply_clahe

output_label_preprocessing_function: Callable[
    [np.ndarray], np.ndarray
] = lambda _img: img_to_minmax(_img, 127, (0, 255))


class UnetL4ModelHelper(ModelHelper):
    def __init__(
        self,
        model_descriptor: ModelDescriptor = unet_l4_model_descriptor_default,
        alpha: float = 1.0,
    ):
        super().__init__(model_descriptor, alpha)

    def get_model(self) -> tf.keras.models.Model:
        return unet_l4(
            input_name=self.model_descriptor.inputs[0][0],
            input_shape=self.model_descriptor.inputs[0][1],
            output_name=self.model_descriptor.outputs[0][0],
            alpha=self.alpha,
        )

    def compile_model(
        self,
        model: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(lr=1e-4),
        loss_list: List[LossDescriptor] = unet_l4_loss_descriptors_default,
        metrics: List[tf.keras.metrics.Metric] = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            BinaryClassMeanIoU(name="mean_iou"),
        ],
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        **kwargs
    ) -> tf.keras.models.Model:
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
