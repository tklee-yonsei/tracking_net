from enum import Enum, unique
from typing import Callable, Optional

import numpy as np
import toolz
from common_py import ArgTypeMixin

from utils.image_transform import gray_image_apply_clahe


@unique
class TrackingTrainingPreProcessing(ArgTypeMixin, Enum):
    none = "none"
    clahe = "clahe"

    def get_preprocess(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """
        사전 정의된 이미지 전처리 함수를 불러옵니다.

        Returns
        -------
        Optional[Callable[[np.ndarray], np.ndarray]]
            `Enum` 옵션에 따른 cv2 이미지 전처리 함수.
        """
        if self == TrackingTrainingPreProcessing.clahe:
            return gray_image_apply_clahe
        else:
            return None


def image_data_generator_shaping(
    cv2_image_transform_function: Callable[[np.ndarray], np.ndarray], before_img
):
    """
    케라스의 `ImageDataGenerator`
    To use `preprocessing_function` of `ImageDataGenerator`.

    Args:
        cv2_image_transform_function (Callable[[np.ndarray], np.ndarray]):
        before_img:

    Returns:
    """
    rr = np.array(before_img, dtype=np.uint8)
    result_img = cv2_image_transform_function(rr)
    return np.reshape(result_img, (result_img.shape[0], result_img.shape[1], 1))


# Image -> Ratio
def gray_image_to_ratio(img: np.ndarray) -> np.ndarray:
    return img / 255.0


def ratio_threshold(ratio_img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    result_ratio_img = ratio_img.copy()
    result_ratio_img[result_ratio_img > threshold] = 1
    result_ratio_img[result_ratio_img <= threshold] = 0
    return result_ratio_img


def gray_image_to_threshold_ratio(
    img: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    return toolz.pipe(
        img,
        gray_image_to_ratio,
        lambda ratio_img: ratio_threshold(ratio_img, threshold),
    )
