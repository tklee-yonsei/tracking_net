from typing import Callable, Generator, List, Tuple

import keras
import numpy as np
import toolz
from image_keras.custom.metrics import BinaryClassMeanIoU
from image_keras.model_manager import LossDescriptor, ModelDescriptor, ModelHelper
from image_keras.utils.image_color_transform import (
    color_map_generate,
    image_detach_with_id_color_probability_list,
)
from image_keras.utils.image_info import get_all_colors
from image_keras.utils.image_transform import (
    InterpolationEnum,
    gray_image_apply_clahe,
    img_resize,
    img_to_minmax,
)
from keras.metrics import Metric
from keras.models import Model
from keras.optimizers import Adam, Optimizer
from models.color_tracking.model_006.model import model_006

bin_size: int = 30

model006_model_descriptor_default: ModelDescriptor = ModelDescriptor(
    inputs=[
        ("main_image", (256, 256, 1)),
        ("ref_image", (256, 256, 1)),
        ("bin1_label", (64, 64, bin_size)),
        ("bin2_label", (128, 128, bin_size)),
        ("bin3_label", (256, 256, bin_size)),
    ],
    outputs=[("output", (256, 256, bin_size))],
)

model006_loss_descriptors_default: List[LossDescriptor] = [
    LossDescriptor(loss=keras.losses.CategoricalCrossentropy(), weight=1.0)
]

input_main_image_preprocessing_function: Callable[
    [np.ndarray], np.ndarray
] = gray_image_apply_clahe

input_ref_image_preprocessing_function: Callable[
    [np.ndarray], np.ndarray
] = gray_image_apply_clahe


def generate_color_map(
    img: np.ndarray, zero_first: bool = True, add_black_if_not_exist: bool = False
) -> List[Tuple[int, Tuple[int, int, int]]]:
    img_color_num: List[Tuple[Tuple[int, int, int], int]] = get_all_colors(
        img, zero_first
    )[:bin_size]
    img_color_list: List[Tuple[int, int, int]] = list(
        map(lambda el: el[0], img_color_num)
    )
    return color_map_generate(img_color_list, add_black_if_not_exist)


def input_ref1_label_preprocessing_function(label: np.ndarray) -> np.ndarray:
    id_color_list = generate_color_map(label, True, True)
    return image_detach_with_id_color_probability_list(
        label, id_color_list, bin_size, 2
    )


def input_ref2_label_preprocessing_function(label: np.ndarray) -> np.ndarray:
    id_color_list = generate_color_map(label, True, True)
    return image_detach_with_id_color_probability_list(
        label, id_color_list, bin_size, 1
    )


def input_ref3_label_preprocessing_function(label: np.ndarray) -> np.ndarray:
    id_color_list = generate_color_map(label, True, True)
    return image_detach_with_id_color_probability_list(
        label, id_color_list, bin_size, 0
    )


def output_label_preprocessing_function(
    label: np.ndarray, ref3_label: np.ndarray
) -> np.ndarray:
    ref3_id_color_list = generate_color_map(ref3_label, True, True)
    return image_detach_with_id_color_probability_list(
        label, ref3_id_color_list, bin_size, 0
    )


class Model006ModelHelper(ModelHelper):
    def __init__(
        self,
        pre_trained_unet_l4_model: Model,
        model_descriptor: ModelDescriptor = model006_model_descriptor_default,
        alpha: float = 1.0,
    ):
        super().__init__(model_descriptor, alpha)
        self.pre_trained_unet_l4_model = pre_trained_unet_l4_model

    def get_model(self) -> Model:
        return model_006(
            pre_trained_unet_l4_model=self.pre_trained_unet_l4_model,
            input_main_image_name=self.model_descriptor.inputs[0][0],
            input_main_image_shape=self.model_descriptor.inputs[0][1],
            input_ref_image_name=self.model_descriptor.inputs[1][0],
            input_ref_image_shape=self.model_descriptor.inputs[1][1],
            input_ref1_label_name=self.model_descriptor.inputs[2][0],
            input_ref1_label_shape=self.model_descriptor.inputs[2][1],
            input_ref2_label_name=self.model_descriptor.inputs[3][0],
            input_ref2_label_shape=self.model_descriptor.inputs[3][1],
            input_ref3_label_name=self.model_descriptor.inputs[4][0],
            input_ref3_label_shape=self.model_descriptor.inputs[4][1],
            output_name=self.model_descriptor.outputs[0][0],
            bin_num=bin_size,
            alpha=self.alpha,
        )

    def compile_model(
        self,
        model: Model,
        optimizer: Optimizer = Adam(lr=1e-4),
        loss_list: List[LossDescriptor] = model006_loss_descriptors_default,
        metrics: List[Metric] = [keras.metrics.CategoricalAccuracy(name="acc")],
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


def single_input_main_image_preprocessing(img: np.ndarray) -> np.ndarray:
    resize_to = model006_model_descriptor_default.get_input_sizes()[0]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    return toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_main_image_preprocessing_function,
        processing_for_grayscale_img,
    )(img)


def single_input_ref_image_preprocessing(img: np.ndarray) -> np.ndarray:
    resize_to = model006_model_descriptor_default.get_input_sizes()[0]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    return toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref_image_preprocessing_function,
        processing_for_grayscale_img,
    )(img)


def single_input_ref1_result_preprocessing(img: np.ndarray) -> np.ndarray:
    resize_to = model006_model_descriptor_default.get_input_sizes()[0]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    return toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref1_label_preprocessing_function,
    )(img)


def single_input_ref2_result_preprocessing(img: np.ndarray) -> np.ndarray:
    resize_to = model006_model_descriptor_default.get_input_sizes()[0]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    return toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref2_label_preprocessing_function,
    )(img)


def single_input_ref3_result_preprocessing(img: np.ndarray) -> np.ndarray:
    resize_to = model006_model_descriptor_default.get_input_sizes()[0]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    return toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref3_label_preprocessing_function,
    )(img)


def single_generator(
    main_img: np.ndarray,
    ref_image: np.ndarray,
    ref1_result: np.ndarray,
    ref2_result: np.ndarray,
    ref3_result: np.ndarray,
) -> Generator:
    for index in range(1):
        main_img = np.reshape(main_img, (1,) + main_img.shape)
        ref_image = np.reshape(ref_image, (1,) + ref_image.shape)
        ref1_result = np.reshape(ref1_result, (1,) + ref1_result.shape)
        ref2_result = np.reshape(ref2_result, (1,) + ref2_result.shape)
        ref3_result = np.reshape(ref3_result, (1,) + ref3_result.shape)
        yield [main_img, ref_image, ref1_result, ref2_result, ref3_result]
