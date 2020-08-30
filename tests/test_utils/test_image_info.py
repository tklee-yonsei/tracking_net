import os
from typing import List, Tuple
from unittest import TestCase

import cv2
import numpy as np

from utils import image_info


class TestImageInfo(TestCase):
    current_path: str = os.path.dirname(os.path.abspath(__file__))
    working_path: str = os.getcwd()
    test_path: str = os.path.join(working_path, "tests")
    test_resource_folder_name: str = "test_resources"

    def test_check_grayscale_image(self):
        test_file_name: str = "lenna.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        image = cv2.imread(test_image_fullpath)
        self.assertFalse(image_info.check_grayscale_image(image))
        image = cv2.imread(test_image_fullpath, cv2.IMREAD_GRAYSCALE)
        self.assertTrue(image_info.check_grayscale_image(image))

    def test_get_colors_on_color_image(self):
        test_file_name: str = "lenna.png"
        lenna_colors: int = 148279
        lenna_first_color: np.ndarray = np.array([8, 23, 90])
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        image = cv2.imread(test_image_fullpath)
        self.assertEqual(len(image_info.get_colors_on(image)[0]), lenna_colors)
        self.assertTrue(
            np.array_equal(image_info.get_colors_on(image)[0][0], lenna_first_color)
        )

    def test_get_colors_on_gray_image(self):
        test_file_name: str = "lenna.png"
        lenna_colors: int = 163
        lenna_first_color: np.ndarray = np.array([39])
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        image = cv2.imread(test_image_fullpath, cv2.IMREAD_GRAYSCALE)
        self.assertEqual(len(image_info.get_colors_on(image)[0]), lenna_colors)
        self.assertTrue(
            np.array_equal(image_info.get_colors_on(image)[0][0], lenna_first_color)
        )

    def test_get_all_colors(self):
        test_file_name: str = "lenna.png"
        lenna_colors: int = 148279
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        image = cv2.imread(test_image_fullpath)
        colors_on_image: List[
            Tuple[Tuple[int, int, int], int]
        ] = image_info.get_all_colors(image, True)
        self.assertEqual(len(colors_on_image), lenna_colors)

    def test_get_all_colors__black_color_first(self):
        test_file_name: str = "bone.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        image = cv2.imread(test_image_fullpath)
        colors_on_image: List[
            Tuple[Tuple[int, int, int], int]
        ] = image_info.get_all_colors(image, True)
        self.assertEqual(colors_on_image[0][0], (0, 0, 0))

    def test_get_levels_from_grayscale_image(self):
        test_file_name: str = "bone.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        gray_image = cv2.imread(test_image_fullpath, cv2.IMREAD_GRAYSCALE)
        colors_on_image = image_info.get_all_colors(gray_image, True)
        self.assertEqual(colors_on_image[0][0], (0,))
