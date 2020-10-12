from typing import Callable, List

import numpy as np
from image_keras.custom.metrics import BinaryClassMeanIoU
from image_keras.model_manager import LossDescriptor, ModelDescriptor, ModelHelper
from image_keras.utils.image_transform import gray_image_apply_clahe, img_to_minmax
from models.semantic_segmentation.vanilla_unet.model import vanilla_unet
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer

vanilla_unet_model_descriptor_default: ModelDescriptor = ModelDescriptor(
    inputs=[("input", (572, 572, 1))], outputs=[("output", (386, 386, 1))]
)

vanilla_unet_loss_descriptors_default: List[LossDescriptor] = [
    LossDescriptor(loss=BinaryCrossentropy(), weight=1.0)
]

input_image_preprocessing_function: Callable[
    [np.ndarray], np.ndarray
] = gray_image_apply_clahe

output_label_preprocessing_function: Callable[
    [np.ndarray], np.ndarray
] = lambda _img: img_to_minmax(_img, 127, (0, 255))


class VanillaUnetModelHelper(ModelHelper):
    def __init__(
        self,
        model_descriptor: ModelDescriptor = vanilla_unet_model_descriptor_default,
        alpha: float = 1.0,
    ):
        super().__init__(model_descriptor, alpha)

    def get_model(self) -> Model:
        return vanilla_unet(
            input_name=self.model_descriptor.inputs[0][0],
            input_shape=self.model_descriptor.inputs[0][1],
            output_name=self.model_descriptor.outputs[0][0],
            alpha=self.alpha,
        )

    def compile_model(
        self,
        model: Model,
        optimizer: Optimizer = Adam(lr=1e-4),
        loss_list: List[LossDescriptor] = vanilla_unet_loss_descriptors_default,
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
