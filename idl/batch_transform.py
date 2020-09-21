from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import toolz

from idl.flow_directory import FlowFromDirectory
from utils.optional import get_or_else


def transform_for_batch(
    batch_img: np.ndarray,
    each_image_transform_function: Callable[[np.ndarray], np.ndarray],
    each_transformed_image_save_function_optional: Optional[
        Callable[[int, np.ndarray], None]
    ] = None,
) -> np.ndarray:
    """
    배치 이미지를 각 배치 별로 변환하고, 저장합니다.

    Parameters
    ----------
    batch_img : np.ndarray
        변환할 배치 이미지
    each_image_transform_function : Callable[[np.ndarray], np.ndarray]
        배치 내에서 변환할 함수
    each_transformed_image_save_function_optional : Optional[Callable[[int, int, np.ndarray], None]], optional
        배치 내에서 변환 후, 이미지를 저장할 함수로, 배치 번호 및 이미지를 입력으로 받는 함수, by default None

    Returns
    -------
    np.ndarray
        배치 변환이 완료된 이미지들의 배치
    """
    img_result_list = []
    for i in range(batch_img.shape[0]):
        current_batch_img = batch_img[i, :]
        current_batch_img = current_batch_img.astype(np.uint8)
        current_transformed_img = each_image_transform_function(current_batch_img)
        img_result_list.append(current_transformed_img)
        if each_transformed_image_save_function_optional:
            each_transformed_image_save_function_optional(i, current_transformed_img)
    return np.array(img_result_list)


def generate_iterator_and_transform(
    image_generator: Iterator,
    each_image_transform_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    each_transformed_image_save_function_optional: Optional[
        Callable[[int, int, np.ndarray], None]
    ] = None,
    transform_function_for_all: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Iterator:
    """
    `image_from_directory`에 대해, (선택) (이미지 변환을 적용하고, 변환 결과를 저장한 다음), iterator로 출력

    배치 내 각 개별 이미지에 대해 변환 함수(`batch_transform_function`)가 적용된 후,
    배치 전체에 대한 변환 함수(`transform_function_for_all`)가 적용됩니다.

    Parameters
    ----------
    image_generator : Iterator
        파일을 가져올 `Iterator`
    each_image_transform_function : Optional[Callable[[np.ndarray], np.ndarray]], optional
        배치 내 이미지 변환 함수. 변환 함수가 지정되지 않으면, 변환 없이 그냥 내보냅니다.
    each_transformed_image_save_function_optional : Optional[Callable[[int, int, np.ndarray], None]], optional
        샘플 인덱스 번호, 배치 번호 및 이미지를 입력으로 하는 저장 함수, by default None
    transform_function_for_all : Optional[Callable[[np.ndarray], np.ndarray]], optional
        변환 함수. 배치 전체에 대한 변환 함수

    Yields
    -------
    Iterator
        `image_from_directory`의 generator을 가져와 변환(혹은 변환되지 않은)을 적용한 iterator
    """
    if each_image_transform_function:
        for index, batch_image in enumerate(image_generator):
            each_transformed_image_save_function_optional2 = None
            if each_transformed_image_save_function_optional:
                each_transformed_image_save_function_optional2 = toolz.curry(
                    each_transformed_image_save_function_optional
                )(index)
            batch_image = transform_for_batch(
                batch_img=batch_image,
                each_image_transform_function=get_or_else(
                    each_image_transform_function, lambda v: v
                ),
                each_transformed_image_save_function_optional=each_transformed_image_save_function_optional2,
            )
            if transform_function_for_all:
                batch_image = transform_function_for_all(batch_image)
            yield batch_image
    else:
        for batch_image in image_generator:
            if transform_function_for_all:
                batch_image = transform_function_for_all(batch_image)
            yield batch_image
