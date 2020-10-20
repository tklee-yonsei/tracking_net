import os
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np
import toolz
from image_keras.custom.losses_binary_boundary_crossentropy import (
    BinaryBoundaryCrossentropy,
)
from image_keras.custom.metrics import BinaryClassMeanIoU
from image_keras.model_manager import LossDescriptor, ModelDescriptor, ModelHelper
from image_keras.utils.image_transform import (
    InterpolationEnum,
    gray_image_apply_clahe,
    img_resize,
    img_to_minmax,
    img_to_ratio,
)
from models.semantic_segmentation.unet_l4_002.model import unet_l4
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer

unet_l4_003_model_descriptor_default: ModelDescriptor = ModelDescriptor(
    inputs=[("input", (256, 256, 1))], outputs=[("output", (256, 256, 1))]
)


unet_l4_003_loss_descriptors_default: List[LossDescriptor] = [
    LossDescriptor(loss=BinaryBoundaryCrossentropy(range=3, max=3.0), weight=1.0)
]

input_image_preprocessing_function: Callable[
    [np.ndarray], np.ndarray
] = gray_image_apply_clahe

output_label_preprocessing_function: Callable[
    [np.ndarray], np.ndarray
] = lambda _img: img_to_minmax(_img, 127, (0, 255))


class UnetL4_003ModelHelper(ModelHelper):
    def __init__(
        self,
        model_descriptor: ModelDescriptor = unet_l4_003_model_descriptor_default,
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
        loss_list: List[LossDescriptor] = unet_l4_003_loss_descriptors_default,
        metrics: List[Metric] = [
            BinaryAccuracy(name="accuracy"),
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


# single processing
def processing_for_grayscale_img(img: np.ndarray):
    if len(img.shape) == 2:
        img = np.reshape(img, (img.shape[0], img.shape[1], 1),)
    return img


def single_input_main_image_preprocessing(
    img: np.ndarray, full_path_optional: Optional[Tuple[str, str]] = None
) -> np.ndarray:
    resize_to = unet_l4_003_model_descriptor_default.get_input_sizes()[0]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    img = toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_image_preprocessing_function,
        processing_for_grayscale_img,
    )(img)
    if full_path_optional:
        cv2.imwrite(os.path.join(full_path_optional[0], full_path_optional[1]), img)
    img = img_to_ratio(img)
    return img


def single_generator(main_img: np.ndarray) -> Generator:
    for index in range(1):
        main_img = np.reshape(main_img, (1,) + main_img.shape)
        yield [main_img]

