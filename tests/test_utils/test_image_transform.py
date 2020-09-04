import os
from typing import List, Tuple
from unittest import TestCase

import cv2
import numpy as np

from utils import image_transform


class TestImageTransform(TestCase):
    current_path: str = os.path.dirname(os.path.abspath(__file__))
    working_path: str = os.getcwd()
    test_path: str = os.path.join(working_path, "tests")
    test_resource_folder_name: str = "test_resources"

    def __get_lenna_image(self) -> np.ndarray:
        test_file_name: str = "lenna.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        return cv2.imread(test_image_fullpath)

    def __get_sample_image(self) -> np.ndarray:
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        return cv2.imread(test_image_fullpath)

    def test_img_to_ratio(self):
        """
        `img_to_ratio()` 메서드를 테스트합니다.
        """
        # Prerequisite
        image: np.ndarray = self.__get_sample_image()

        # Processing
        ratio_img: np.ndarray = image_transform.img_to_ratio(image)
        reversed_img: np.ndarray = image_transform.ratio_to_img(ratio_img)
        reversed_img = reversed_img.astype(np.uint8)

        # Result

        # Check
        self.assertTrue(np.array_equal(reversed_img, image))

    def test_img_to_minmax(self):
        # Prerequisite
        image: np.ndarray = self.__get_sample_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Processing
        min_max_img = image_transform.img_to_minmax(image, 127, (0, 255))

        # Result
        num_of_white_pixel: int = 31760

        # Check
        self.assertEqual(np.count_nonzero(min_max_img), num_of_white_pixel)
