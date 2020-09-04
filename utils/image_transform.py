from typing import Tuple

import cv2
import numpy as np


def gray_image_apply_clahe(
    gray_cv2_img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    clahe 필터를 적용합니다. 그레이스케일 이미지에서만 동작합니다.

    Parameters
    ----------
    gray_cv2_img : np.ndarray
        그레이스케일 이미지
    clip_limit : float, optional
        Clip limit, by default 2.0
    tile_grid_size : Tuple[int, int], optional
        타일 그리드 사이즈, by default (8, 8)

    Returns
    -------
    np.ndarray
        clahe 필터를 적용한 이미지
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result_img = clahe.apply(gray_cv2_img)
    return result_img


def img_to_ratio(img: np.ndarray) -> np.ndarray:
    """
    이미지를 비율로 전환합니다.

    Parameters
    ----------
    img : np.ndarray
        비율로 변환할 이미지

    Returns
    -------
    np.ndarray
        비율
    """
    return img / 255.0


def ratio_to_img(ratio_img: np.ndarray) -> np.ndarray:
    """
    비율을 이미지로 전환합니다.

    Parameters
    ----------
    img : np.ndarray
        이미지로 변환할 비율

    Returns
    -------
    np.ndarray
        이미지
    """
    return ratio_img * 255


def img_to_minmax(
    img: np.ndarray, threshold: float = 0.5, min_max: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Threshold를 기준으로, 최소, 최대로 변경합니다.

    Parameters
    ----------
    img : np.ndarray
        이미지
    threshold : float, optional
        Threshold, by default 0.5
    min_max : Tuple[float, float], optional
        최소, 최대 값, by default (0.0, 1.0)

    Returns
    -------
    np.ndarray
        최소, 최대로 변환된 이미지
    """
    result_img = img.copy()
    result_img[result_img > threshold] = min_max[1]
    result_img[result_img <= threshold] = min_max[0]
    return result_img
