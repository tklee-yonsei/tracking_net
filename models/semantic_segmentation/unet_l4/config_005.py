import os
from typing import Callable, Generator, List, Optional, Tuple

import cv2
import numpy as np
import toolz

# from image_keras.custom.losses_binary_boundary_crossentropy import (
#     BinaryBoundaryCrossentropy,
# )
from image_keras.custom.metrics import BinaryClassMeanIoU
from image_keras.model_manager import LossDescriptor, ModelDescriptor, ModelHelper
from image_keras.utils.image_transform import (
    InterpolationEnum,
    gray_image_apply_clahe,
    img_resize,
    img_to_minmax,
    img_to_ratio,
)
from models.semantic_segmentation.unet_l4.model import unet_l4
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Optimizer

unet_l4_model_descriptor_default: ModelDescriptor = ModelDescriptor(
    inputs=[("input", (256, 256, 1))], outputs=[("output", (256, 256, 1))]
)

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper, binary_crossentropy
from tensorflow.python.keras.utils import losses_utils


class BinaryBoundaryCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0,
        reduction=losses_utils.ReductionV2.AUTO,
        range=1,
        max=2.0,
        name="binary_boundary_crossentropy",
    ):
        super(BinaryBoundaryCrossentropy, self).__init__(
            binary_boundary_crossentropy,
            name=name,
            reduction=reduction,
            range=range,
            max=max,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        )
        self.from_logits = from_logits
        self.range = range
        self.max = max


def binary_boundary_crossentropy(
    y_true,
    y_pred,
    from_logits=False,
    label_smoothing=0,
    range: int = 1,
    max: float = 2.0,
):
    """
    [summary]

    Parameters
    ----------
    y_true : [type]
        [description]
    y_pred : [type]
        [description]
    from_logits : bool, optional
        [description], by default False
    label_smoothing : int, optional
        [description], by default 0
    range : int, optional
        [description], by default 0
    max : float, optional
        [description], by default 1.0

    Returns
    -------
    [type]
        [description]

    Examples
    --------
    >>> from image_keras.custom.losses_binary_boundary_crossentropy import binary_boundary_crossentropy
    >>> import cv2
    >>> a = cv2.imread("tests/test_resources/a.png", cv2.IMREAD_GRAYSCALE)
    >>> a_modified = (a / 255).reshape(1, a.shape[0], a.shape[1], 1)
    >>> binary_boundary_crossentropy(a_modified, a_modified, range=1, max=2)
    """
    bce = binary_crossentropy(
        y_true=y_true,
        y_pred=y_pred,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
    )

    def count_around_blocks(arr, range: int = 1):
        ones = tf.ones_like(arr)
        size = range * 2 + 1
        if range < 1:
            size = 1
        extracted = tf.image.extract_patches(
            images=ones,
            sizes=[1, size, size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        result = tf.reduce_sum(extracted, axis=-1)
        if range > 0:
            result -= 1
        return result

    def count_around_blocks2(arr, range: int = 1):
        size = range * 2 + 1
        if range < 1:
            size = 1
        extracted = tf.image.extract_patches(
            images=arr,
            sizes=[1, size, size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        e_base = extracted[:, :, :, tf.shape(extracted)[-1] // 2]
        e_base = tf.reshape(e_base, (-1, tf.shape(arr)[1], tf.shape(arr)[2], 1))
        e_base_expanded = tf.reshape(
            tf.repeat(e_base, tf.shape(extracted)[-1]),
            (-1, tf.shape(arr)[1], tf.shape(arr)[2], tf.shape(extracted)[-1]),
        )
        same_values = tf.math.equal(extracted, e_base_expanded)
        result_1 = tf.shape(extracted)[-1] - tf.cast(
            tf.math.count_nonzero(same_values, axis=-1), tf.int32
        )
        result_1 = tf.reshape(result_1, (-1, tf.shape(arr)[1], tf.shape(arr)[2], 1))
        result_1 = tf.cast(result_1, tf.float32)
        result_1 += arr
        block_counts = tf.reshape(
            count_around_blocks(arr, range), (-1, tf.shape(arr)[1], tf.shape(arr)[2], 1)
        )
        modify_result_1 = -(size ** 2 - block_counts)
        modify_result_1 = modify_result_1 * arr
        modify_result_1 = tf.cast(modify_result_1, tf.float32)
        diff_block_count = result_1 + modify_result_1
        return diff_block_count

    around_block_count = count_around_blocks(y_true, range=range)
    around_block_count = tf.reshape(
        around_block_count, (-1, tf.shape(y_true)[1], tf.shape(y_true)[2], 1)
    )
    around_block_count = tf.cast(around_block_count, tf.float32)

    diff_block_count = count_around_blocks2(y_true, range=range)
    diff_block_count = tf.reshape(
        diff_block_count, (-1, tf.shape(y_true)[1], tf.shape(y_true)[2], 1)
    )
    diff_block_count = tf.cast(diff_block_count, tf.float32)

    diff_ratio = diff_block_count / around_block_count

    importance_of_area = tf.math.count_nonzero(
        tf.greater_equal(diff_ratio, 0.5), axis=-1
    )
    importance_within_area = (tf.shape(y_true)[1] * tf.shape(y_true)[2]) / (
        importance_of_area + 1
    )
    importance_of_area = tf.cast(importance_of_area, tf.float32)
    importance_of_area = tf.reshape(
        importance_of_area, (-1, tf.shape(y_true)[1], tf.shape(y_true)[2], 1)
    )

    # importance_within_area = tf.cast(
    #     (tf.shape(y_true)[1] * tf.shape(y_true)[2]), tf.float32
    # ) / (importance_of_area + 1e-7)

    diff_ratio = diff_ratio * importance_of_area

    diff_ratio = 1.0 + diff_ratio
    # diff_ratio = 1.0 + tf.cast(tf.math.maximum(max - 1.0, 0), tf.float32) * diff_ratio
    diff_ratio = tf.reshape(diff_ratio, (-1, tf.shape(y_true)[1], tf.shape(y_true)[2]))

    return bce * diff_ratio


unet_l4_loss_descriptors_default: List[LossDescriptor] = [
    LossDescriptor(loss=BinaryBoundaryCrossentropy(range=5, max=100.0), weight=1.0)
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
    resize_to = unet_l4_model_descriptor_default.get_input_sizes()[0]
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

