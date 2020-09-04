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
