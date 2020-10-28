import math
import os
from random import Random
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import toolz
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
    img_to_ratio,
)
from models.ref_local_tracking.ref_local_tracking_model_002.model import (
    ref_local_tracking_model_002,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer

bin_size: int = 30

ref_local_tracking_model_002_model_descriptor_default: ModelDescriptor = ModelDescriptor(
    inputs=[
        ("main_image", (256, 256, 1)),
        ("ref_image", (256, 256, 1)),
        ("bin_label_1", (32, 32, bin_size)),
        ("bin_label_2", (64, 64, bin_size)),
        ("bin_label_3", (128, 128, bin_size)),
        ("bin_label_4", (256, 256, bin_size)),
    ],
    outputs=[("output", (256, 256, bin_size))],
)

model002_loss_descriptors_default: List[LossDescriptor] = [
    LossDescriptor(loss=CategoricalCrossentropy(), weight=1.0)
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


def input_ref_label_1_preprocessing_function(label: np.ndarray) -> np.ndarray:
    id_color_list = generate_color_map(label, True, True)
    return image_detach_with_id_color_probability_list(
        label, id_color_list, bin_size, 3
    )


def input_ref_label_2_preprocessing_function(label: np.ndarray) -> np.ndarray:
    id_color_list = generate_color_map(label, True, True)
    return image_detach_with_id_color_probability_list(
        label, id_color_list, bin_size, 2
    )


def input_ref_label_3_preprocessing_function(label: np.ndarray) -> np.ndarray:
    id_color_list = generate_color_map(label, True, True)
    return image_detach_with_id_color_probability_list(
        label, id_color_list, bin_size, 1
    )


def input_ref_label_4_preprocessing_function(label: np.ndarray) -> np.ndarray:
    id_color_list = generate_color_map(label, True, True)
    return image_detach_with_id_color_probability_list(
        label, id_color_list, bin_size, 0
    )


def output_label_preprocessing_function(
    label: np.ndarray, ref_label: np.ndarray
) -> np.ndarray:
    ref_id_color_list = generate_color_map(ref_label, True, True)
    return image_detach_with_id_color_probability_list(
        label, ref_id_color_list, bin_size, 0
    )


class RefModel002ModelHelper(ModelHelper):
    def __init__(
        self,
        pre_trained_unet_l4_model: Model,
        model_descriptor: ModelDescriptor = ref_local_tracking_model_002_model_descriptor_default,
        alpha: float = 1.0,
    ):
        super().__init__(model_descriptor, alpha)
        self.pre_trained_unet_l4_model = pre_trained_unet_l4_model

    def get_model(self) -> Model:
        return ref_local_tracking_model_002(
            pre_trained_unet_l4_model=self.pre_trained_unet_l4_model,
            input_main_image_name=self.model_descriptor.inputs[0][0],
            input_main_image_shape=self.model_descriptor.inputs[0][1],
            input_ref_image_name=self.model_descriptor.inputs[1][0],
            input_ref_image_shape=self.model_descriptor.inputs[1][1],
            input_ref_label_1_name=self.model_descriptor.inputs[2][0],
            input_ref_label_1_shape=self.model_descriptor.inputs[2][1],
            input_ref_label_2_name=self.model_descriptor.inputs[3][0],
            input_ref_label_2_shape=self.model_descriptor.inputs[3][1],
            input_ref_label_3_name=self.model_descriptor.inputs[4][0],
            input_ref_label_3_shape=self.model_descriptor.inputs[4][1],
            input_ref_label_4_name=self.model_descriptor.inputs[5][0],
            input_ref_label_4_shape=self.model_descriptor.inputs[5][1],
            output_name=self.model_descriptor.outputs[0][0],
            bin_num=bin_size,
            alpha=self.alpha,
        )

    def compile_model(
        self,
        model: Model,
        optimizer: Optimizer = Adam(lr=1e-4),
        loss_list: List[LossDescriptor] = model002_loss_descriptors_default,
        metrics: List[Metric] = [CategoricalAccuracy(name="acc")],
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


class RefTrackingSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_file_names: List[str],
        main_image_folder_name: str,
        ref_image_folder_name: str,
        ref_result_label_folder_name: str,
        output_label_folder_name: str,
        ref_result1_preprocessing_function: Callable[
            [np.ndarray], np.ndarray
        ] = input_ref_label_1_preprocessing_function,
        ref_result2_preprocessing_function: Callable[
            [np.ndarray], np.ndarray
        ] = input_ref_label_2_preprocessing_function,
        ref_result3_preprocessing_function: Callable[
            [np.ndarray], np.ndarray
        ] = input_ref_label_3_preprocessing_function,
        ref_result4_preprocessing_function: Callable[
            [np.ndarray], np.ndarray
        ] = input_ref_label_4_preprocessing_function,
        output_label_preprocessing_function: Callable[
            [np.ndarray, np.ndarray], np.ndarray
        ] = output_label_preprocessing_function,
        main_image_preprocessing_function: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = toolz.compose_left(input_main_image_preprocessing_function, img_to_ratio),
        ref_image_preprocessing_function: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = toolz.compose_left(input_ref_image_preprocessing_function, img_to_ratio),
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.image_file_names = image_file_names
        self.main_image_folder_name = main_image_folder_name
        self.ref_image_folder_name = ref_image_folder_name
        self.ref_result_label_folder_name = ref_result_label_folder_name
        self.output_label_folder_name = output_label_folder_name
        self.main_image_preprocessing_function = main_image_preprocessing_function
        self.ref_image_preprocessing_function = ref_image_preprocessing_function
        self.ref_result1_preprocessing_function = ref_result1_preprocessing_function
        self.ref_result2_preprocessing_function = ref_result2_preprocessing_function
        self.ref_result3_preprocessing_function = ref_result3_preprocessing_function
        self.ref_result4_preprocessing_function = ref_result4_preprocessing_function
        self.output_label_preprocessing_function = output_label_preprocessing_function
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        if shuffle is True:
            Random(seed).shuffle(self.image_file_names)

    def __len__(self):
        return math.ceil(len(self.image_file_names) / self.batch_size)

    def __getitem__(self, idx):
        batch_image_file_names = self.image_file_names[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_main_images = []
        batch_ref_images = []
        batch_ref1_results = []
        batch_ref2_results = []
        batch_ref3_results = []
        batch_ref4_results = []
        batch_output_labels = []
        for image_file_name in batch_image_file_names:
            main_img = cv2.imread(
                os.path.join(self.main_image_folder_name, image_file_name),
                cv2.IMREAD_GRAYSCALE,
            )
            if self.main_image_preprocessing_function is not None:
                main_img = self.main_image_preprocessing_function(main_img)
            batch_main_images.append(main_img)

            ref_img = cv2.imread(
                os.path.join(self.ref_image_folder_name, image_file_name),
                cv2.IMREAD_GRAYSCALE,
            )
            if self.ref_image_preprocessing_function is not None:
                ref_img = self.ref_image_preprocessing_function(ref_img)
            batch_ref_images.append(ref_img)

            ref_result_label = cv2.imread(
                os.path.join(self.ref_result_label_folder_name, image_file_name)
            )

            ref_result1 = self.ref_result1_preprocessing_function(ref_result_label)
            batch_ref1_results.append(ref_result1)

            ref_result2 = self.ref_result2_preprocessing_function(ref_result_label)
            batch_ref2_results.append(ref_result2)

            ref_result3 = self.ref_result3_preprocessing_function(ref_result_label)
            batch_ref3_results.append(ref_result3)

            ref_result4 = self.ref_result4_preprocessing_function(ref_result_label)
            batch_ref4_results.append(ref_result4)

            output_label = cv2.imread(
                os.path.join(self.output_label_folder_name, image_file_name)
            )
            output_label = self.output_label_preprocessing_function(
                output_label, ref_result_label,
            )
            batch_output_labels.append(output_label)

        X = [
            np.array(batch_main_images),
            np.array(batch_ref_images),
            np.array(batch_ref1_results),
            np.array(batch_ref2_results),
            np.array(batch_ref3_results),
            np.array(batch_ref4_results),
        ]
        Y = [np.array(batch_output_labels)]

        return (X, Y)


# single processing
def processing_for_grayscale_img(img: np.ndarray):
    if len(img.shape) == 2:
        img = np.reshape(img, (img.shape[0], img.shape[1], 1),)
    return img


def single_input_main_image_preprocessing(
    img: np.ndarray, full_path_optional: Optional[Tuple[str, str]] = None
) -> np.ndarray:
    resize_to = ref_local_tracking_model_002_model_descriptor_default.get_input_sizes()[
        1
    ]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    img = toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_main_image_preprocessing_function,
        processing_for_grayscale_img,
    )(img)
    if full_path_optional:
        cv2.imwrite(os.path.join(full_path_optional[0], full_path_optional[1]), img)
    img = img_to_ratio(img)
    return img


def single_input_ref_image_preprocessing(
    img: np.ndarray, full_path_optional: Optional[Tuple[str, str]] = None
) -> np.ndarray:
    resize_to = ref_local_tracking_model_002_model_descriptor_default.get_input_sizes()[
        0
    ]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    img = toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref_image_preprocessing_function,
        processing_for_grayscale_img,
    )(img)
    if full_path_optional:
        cv2.imwrite(os.path.join(full_path_optional[0], full_path_optional[1]), img)
    img = img_to_ratio(img)
    return img


def bin_save(img: np.ndarray, full_path_optional: Optional[Tuple[str, str]] = None):
    if full_path_optional:
        for i in range(img.shape[2]):
            img_name = full_path_optional[1][
                : full_path_optional[1].rfind(".")
            ] + "_{:02d}.png".format(i)
            img_fullpath = os.path.join(full_path_optional[0], img_name)
            cv2.imwrite(img_fullpath, img[:, :, i] * 255)


def single_input_ref_result_1_preprocessing(
    img: np.ndarray, full_path_optional: Optional[Tuple[str, str]] = None
) -> np.ndarray:
    resize_to = ref_local_tracking_model_002_model_descriptor_default.get_input_sizes()[
        0
    ]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    img = toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref_label_1_preprocessing_function,
    )(img)
    bin_save(img, full_path_optional)
    return img


def single_input_ref_result_2_preprocessing(
    img: np.ndarray, full_path_optional: Optional[Tuple[str, str]] = None
) -> np.ndarray:
    resize_to = ref_local_tracking_model_002_model_descriptor_default.get_input_sizes()[
        0
    ]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    img = toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref_label_2_preprocessing_function,
    )(img)
    bin_save(img, full_path_optional)
    return img


def single_input_ref_result_3_preprocessing(
    img: np.ndarray, full_path_optional: Optional[Tuple[str, str]] = None
) -> np.ndarray:
    resize_to = ref_local_tracking_model_002_model_descriptor_default.get_input_sizes()[
        0
    ]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    img = toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref_label_3_preprocessing_function,
    )(img)
    bin_save(img, full_path_optional)
    return img


def single_input_ref_result_4_preprocessing(
    img: np.ndarray, full_path_optional: Optional[Tuple[str, str]] = None
) -> np.ndarray:
    resize_to = ref_local_tracking_model_002_model_descriptor_default.get_input_sizes()[
        0
    ]
    resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest
    img = toolz.compose_left(
        lambda _img: img_resize(_img, resize_to, resize_interpolation),
        input_ref_label_4_preprocessing_function,
    )(img)
    bin_save(img, full_path_optional)
    return img


def single_generator(
    main_img: np.ndarray,
    ref_image: np.ndarray,
    ref_result_1: np.ndarray,
    ref_result_2: np.ndarray,
    ref_result_3: np.ndarray,
    ref_result_4: np.ndarray,
) -> Generator:
    for index in range(1):
        main_img = np.reshape(main_img, (1,) + main_img.shape)
        ref_image = np.reshape(ref_image, (1,) + ref_image.shape)
        ref_result_1 = np.reshape(ref_result_1, (1,) + ref_result_1.shape)
        ref_result_2 = np.reshape(ref_result_2, (1,) + ref_result_2.shape)
        ref_result_3 = np.reshape(ref_result_3, (1,) + ref_result_3.shape)
        ref_result_4 = np.reshape(ref_result_4, (1,) + ref_result_4.shape)
        yield [
            main_img,
            ref_image,
            ref_result_1,
            ref_result_2,
            ref_result_3,
            ref_result_4,
        ]

