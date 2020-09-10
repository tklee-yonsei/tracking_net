from typing import Tuple

import numpy as np


def count_bgr_color_pixels(
    color_img: np.ndarray, bgr_color: Tuple[int, int, int] = [0, 0, 0]
) -> int:
    return np.count_nonzero(np.all(color_img == bgr_color, axis=2))
