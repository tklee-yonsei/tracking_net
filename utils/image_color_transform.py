from typing import List, Tuple

import numpy as np

from utils.image_info import get_all_colors


def color_map_generate(
    colors: List[Tuple[int, int, int]], add_black_if_not_exist: bool = False
) -> List[Tuple[int, Tuple[int, int, int]]]:
    """
    컬러 리스트에서 [(id, 컬러)] 리스트를 생성합니다.
    
    단, `(0, 0, 0)`이 있을 경우, 검은색의 id는 0으로 할당됩니다.
    `add_black_if_not_exist` 옵션에 따라, 검은 색이 리스트에 없는 경우에도 0번으로 할당 가능합니다.

    Parameters
    ----------
    colors : List[Tuple[int, int, int]]
        BGR 컬러의 리스트
    add_black_if_not_exist : bool
        만약, 컬러 리스트에 black이 없다면, black을 추가합니다.

    Returns
    -------
    List[Tuple[int, Tuple[int, int, int]]]
        ID 및 BGR 컬러의 리스트
    """
    code_tuples: List[Tuple[int, Tuple[int, int, int]]] = []
    black_exists: bool = False
    if (0, 0, 0) in colors:
        colors_without_black = list(filter(lambda el: el != (0, 0, 0), colors))
        code_tuples.append((0, (0, 0, 0)))
        black_exists = True
    else:
        colors_without_black = colors
        if add_black_if_not_exist:
            code_tuples.append((0, (0, 0, 0)))
            black_exists = True
    for index, color in enumerate(colors_without_black):
        new_index = index + 1 if black_exists else index
        code_tuples.append((new_index, color))
    return code_tuples


def image_detach(color_img: np.ndarray, bin_num: int) -> np.ndarray:
    """
    컬러 이미지 `color_img`를 색상에 따른 각 인스턴스 객체로 분리합니다.

    Parameters
    ----------
    color_img : np.ndarray
        컬러 이미지
    bin_num : int
        분리할 최대 인스턴스 객체의 개수

    Returns
    -------
    np.ndarray
        `color_img`의 width, height에 `bin_num` channel인 `np.ndarray`
    
    Examples
    --------
    >>> import cv2
    >>> from utils.image_transform import image_detach
    >>> img = cv2.imread("../simple_image_generator/samples/label.png")
    >>> image_detach(img, 30)
    """
    color_list: List[Tuple[int, int, int]] = list(
        map(lambda v: v[0], get_all_colors(color_img, True))
    )
    id_color_list: List[Tuple[int, Tuple[int, int, int]]] = color_map_generate(
        color_list
    )
    return image_detach_with_id_color_list(color_img, id_color_list, bin_num)


def image_detach_with_id_color_list(
    color_img: np.ndarray,
    id_color_list: List[Tuple[int, Tuple[int, int, int]]],
    bin_num: int,
    mask_value: float = 1.0,
) -> np.ndarray:
    """
    컬러 이미지 `color_img`를 색상에 따른 각 인스턴스 객체로 분리합니다.
    

    Parameters
    ----------
    color_img : np.ndarray
        컬러 이미지
    id_color_list : List[Tuple[int, Tuple[int, int, int]]]
        ID, BGR 컬러 튜플의 리스트. `(0, 0, 0)`이 있는 경우, `(0, 0, 0)`이 맨 앞에 옵니다.
    bin_num : int
        분리할 최대 인스턴스 객체의 개수
    mask_value : float
        색상이 존재하는 이미지의 객체에 덮어 씌울 값, by default 1.

    Returns
    -------
    np.ndarray
        `color_img`의 width, height에 `bin_num` channel인 `np.ndarray`

    """
    result: np.ndarray = np.zeros((color_img.shape[0], color_img.shape[1], bin_num))
    for index, id_color in enumerate(id_color_list):
        result_image = np.zeros(
            (color_img.shape[0], color_img.shape[1], 1), dtype=np.float32
        )
        mask = np.all(color_img == id_color[1], axis=-1)
        result_image[mask] = mask_value
        result[:, :, index : index + 1] = result_image
    return result


def image_detach_with_id_color_probability_list(
    color_img: np.ndarray,
    id_color_list: List[Tuple[int, Tuple[int, int, int]]],
    bin_num: int,
    resize_by_power_of_two: int = 0,
) -> np.ndarray:
    """
    컬러 이미지 `color_img`를 색상에 따른 각 인스턴스 확률 객체로 분리합니다.

    Parameters
    ----------
    color_img : np.ndarray
        컬러 이미지
    id_color_list : List[Tuple[int, Tuple[int, int, int]]]
        ID, BGR 컬러 튜플의 리스트. `(0, 0, 0)`이 있는 경우, `(0, 0, 0)`이 맨 앞에 옵니다.
    bin_num : int
        분리할 최대 인스턴스 객체의 개수
    resize_by_power_of_two : int
        줄일 이미지의 사이즈. 2의 제곱값. 0이면 그대로. 
        예를 들어, 1이면, 1/2로 크기를 줄이고, 2이면, 1/4로 크기를 줄입니다. by default 0.

    Returns
    -------
    np.ndarray
        `color_img`의 width, height에 `bin_num` channel인 `np.ndarray`

    Examples
    --------
    >>> import cv2
    >>> test_img = cv2.imread("../../../Downloads/test.png")
    >>> from utils import image_transform
    >>> test_img_colors = image_transform.get_rgb_color_cv2(test_img, exclude_black=False)
    >>> test_img_color_map = image_transform.color_map_generate(test_img_colors)
    >>> detached_image = image_transform.image_detach_with_id_color_list(test_img, test_img_color_map, 30, 1)
    >>> detached_probability_image_0 = image_transform.image_detach_with_id_color_probability_list(test_img, test_img_color_map, 30, 0)
    >>> detached_probability_image_1 = image_transform.image_detach_with_id_color_probability_list(test_img, test_img_color_map, 30, 1)
    >>> detached_probability_image_2 = image_transform.image_detach_with_id_color_probability_list(test_img, test_img_color_map, 30, 2)
    >>> detached_probability_image_3 = image_transform.image_detach_with_id_color_probability_list(test_img, test_img_color_map, 30, 3)
    """
    result: np.ndarray = image_detach_with_id_color_list(
        color_img, id_color_list, bin_num, 1.0
    )
    ratio = 2 ** resize_by_power_of_two

    result_height, result_width, _ = result.shape
    result2 = shrink3D(result, result_height // ratio, result_width // ratio, bin_num)
    result2 = np.divide(result2, ratio ** 2)
    return result2


def shrink(data: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    2D 배열을 요약합니다. 

    Parameters
    ----------
    data : np.ndarray
        요약할 2D numpy 형태의 데이터
    rows : int
        반환할 행의 수
    cols : int
        반환할 열의 수

    Returns
    -------
    np.ndarray
        요약된 2D numpy 형태의 데이터

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[ 1, 2, 3, 4],
            [ 5 ,6, 7, 8],
            [ 9,10,11,12],
            [13,14,15,16]])
    >>> shrink(a, 2, 2)
    array([[14, 22],    # [[1+2+5+6, 3+4+7+8],
        [46, 54]])      #  [9+10+13+14, 11+12+15+16]]
    >>> shrink(a, 2, 1)
    array([[ 36],       # [[1+2+5+6+3+4+7+8],
        [100]])         #  [9+10+13+14+11+12+15+16]]
    """
    return (
        data.reshape(rows, data.shape[0] // rows, cols, data.shape[1] // cols)
        .sum(axis=1)
        .sum(axis=2)
    )


def shrink3D(data: np.ndarray, rows: int, cols: int, channels: int) -> np.ndarray:
    """
    3D 배열을 요약합니다. 

    Parameters
    ----------
    data : np.ndarray
        요약할 3D numpy 형태의 데이터
    rows : int
        반환할 행의 수
    cols : int
        반환할 열의 수
    channels : int
        반환할 채널의 수

    Returns
    -------
    np.ndarray
        요약된 2D numpy 형태의 데이터

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[[ 1, 2], [3, 4]],
            [ [5 ,6],  [7, 8]],
            [ [9,10],  [11,12]],
            [ [13,14], [15,16]]])
    >>> shrink3D(a,2,1,2)
        array([[[16, 20]],  # [[[   1+3+5+7, 2+4+6+8]],
            [[48, 52]]])    #  [[9+11+13+15, 10+12+14+16]]]
    >>> shrink3D(a,2,1,1)
        array([[[ 36]],     # [[[   1+3+5+7+2+4+6+8]],
            [[100]]])       #  [[9+11+13+15+10+12+14+16]]]
    """
    return (
        data.reshape(
            rows,
            data.shape[0] // rows,
            cols,
            data.shape[1] // cols,
            channels,
            data.shape[2] // channels,
        )
        .sum(axis=1)
        .sum(axis=2)
        .sum(axis=3)
    )
