from typing import List, Tuple

import numpy as np


def check_grayscale_image(img: np.ndarray) -> bool:
    """
    Open CV 이미지의 그레이스케일 이미지 여부를 체크합니다.

    Parameters
    ----------
    img : np.ndarray
        cv2 이미지

    Returns
    -------
    bool
        그레이스케일 이미지 여부
    """
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def get_colors_on(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Open CV 이미지에서 사용된 컬러 및 개수를 반환합니다.
    그레이스케일 이미지의 경우, 색상 레벨(0-255) 및 개수를 반환합니다.

    Parameters
    ----------
    img : np.ndarray
        사용된 컬러를 분석할 cv2 이미지

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        이미지에서 사용된 컬러 및 총 개수 튜플
        (array([[  0,   0,   0], ..., [230,  82, 193]], dtype=uint8), array([298, ...,  80]))
        (array([[  0], ..., [230]], dtype=uint8), array([298, ...,  80]))

    Examples
    --------
    >>> img = cv2.imread("test.png")
    >>> img.shape
    (32, 32, 3)
    >>> get_colors_from_bgr_image(img)
    (array([[  0,   0,   0],
        [ 32, 107, 219],
        [ 33, 217, 245],
        [ 54,  54, 211],
        [154, 235, 110],
        [220, 136, 136],
        [230,  82, 193]], dtype=uint8), array([298, 119, 181, 140, 150,  56,  80]))
    >>> img2 = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    >>> img2.shape
    (32, 32)
    >>> get_levels_from_grayscale_image(img2)
    (array([[  0], [100], [131], [132], [145], [188], [204]], dtype=uint8), array([298, 140, 119,  80,  56, 150, 181]))
    """
    color_channel = 1 if check_grayscale_image(img) else img.shape[2]
    return np.unique(img.reshape(-1, color_channel), axis=0, return_counts=True)


def get_all_colors(
    img: np.ndarray, zero_first: bool = False
) -> List[Tuple[Tuple[int, int, int], int]]:
    """
    Open CV 컬러 이미지에서 사용된 컬러 및 개수를 반환합니다.
    
    `zero_first`가 `True`일 경우, 검은색을 맨 앞으로 이동합니다.
    `zero_first`가 `True`이더라도, 검은색이 없다면, 맨 앞은 검은색이 아닌 다른 색상입니다.

    Parameters
    ----------
    img : np.ndarray
        사용된 컬러를 분석할 cv2 이미지(BGR or 그레이스케일)
    zero_first : bool
        검은색(`(0, 0, 0)`, 그레이스케일일 경우, `(0, )`)을 맨 앞으로 이동합니다. by defaults False.

    Returns
    -------
    List[Tuple[Tuple[int, int, int], int]]
        이미지에서 사용된 컬러(BGR) or 색상 레벨(그레이스케일) 및 총 개수 튜플의 리스트

    Examples
    --------
    >>> img = cv2.imread("test.png")
    >>> img.shape
    (32, 32, 3)
    >>> get_all_colors(img, zero_first=True)
        [((0, 0, 0), 11272),
        ((64, 255, 0), 23679),
        ((195, 187, 90), 8256),
        ((36, 153, 187), 6437),
        ((128, 128, 0), 4466),
        ((73, 243, 229), 2094),
        ((236, 159, 0), 1507),
        ((39, 91, 217), 1494),
        ((170, 55, 92), 1422),
        ((120, 164, 75), 1254),
        ((86, 34, 148), 1083),
        ((18, 209, 220), 881),
        ((97, 23, 196), 508),
        ((0, 0, 255), 376),
        ((166, 151, 20), 344),
        ((137, 24, 193), 257),
        ((235, 90, 150), 206)]
    >>> img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
    >>> img.shape
    (32, 32)
    >>> get_all_colors(img, zero_first=True)
        [((0, ), 11272),
        ...
        ((235, ), 206)]
    """
    black_color = 0 if check_grayscale_image(img) else (0, 0, 0)
    image_colors = get_colors_on(img)
    colors_counts_tuple = list(map(tuple, image_colors[0])), image_colors[1]
    color_count_tuples = list(zip(colors_counts_tuple[0], colors_counts_tuple[1]))
    color_count_tuples = sorted(color_count_tuples, key=lambda x: x[1], reverse=True)
    black_color_indexes = [
        i for i, v in enumerate(color_count_tuples) if v[0] == black_color
    ]
    if len(black_color_indexes) > 0 and zero_first:
        zero_value = color_count_tuples[black_color_indexes[0]]
        color_count_tuples = (
            [zero_value]
            + color_count_tuples[: black_color_indexes[0]]
            + color_count_tuples[black_color_indexes[0] + 1 :]
        )
    return color_count_tuples
