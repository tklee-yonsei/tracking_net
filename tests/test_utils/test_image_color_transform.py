import os
from typing import List, Tuple
from unittest import TestCase

import cv2
import numpy as np

from utils import image_color_transform, image_info


class TestImageColorTransform(TestCase):
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

    def test_color_map_generate_for_add_black(self):
        """
        `color_map_generate()` 메서드를 테스트합니다.

        검은 색이 없는 경우, 검은 색을 맨 앞으로 추가하는 옵션을 테스트합니다.
        """
        # Prerequisite
        image: np.ndarray = self.__get_lenna_image()
        image_colors: List[Tuple[int, int, int]] = list(
            map(lambda v: v[0], image_info.get_all_colors(image, True))
        )

        # Processing
        image_color_map: List[
            Tuple[int, Tuple[int, int, int]]
        ] = image_color_transform.color_map_generate(image_colors, True)

        # Result
        test_result_lenna_color_num: int = 148280
        test_result_lenna_first_color_map: Tuple[Tuple[int, int, int]] = (0, (0, 0, 0))

        # Check
        self.assertEqual(len(image_color_map), test_result_lenna_color_num)
        self.assertEqual(image_color_map[0], test_result_lenna_first_color_map)

    def test_color_map_generate_without_black(self):
        """
        `color_map_generate()` 메서드를 테스트합니다.

        검은 색을 추가하지 않는 옵션을 테스트합니다.
        """
        # Prerequisite
        image: np.ndarray = self.__get_lenna_image()
        image_colors: List[Tuple[int, int, int]] = list(
            map(lambda v: v[0], image_info.get_all_colors(image, True))
        )

        # Processing
        image_color_map: List[
            Tuple[int, Tuple[int, int, int]]
        ] = image_color_transform.color_map_generate(image_colors, False)

        # Result
        test_result_not_lenna_first_color_map: Tuple[Tuple[int, int, int]] = (
            0,
            (0, 0, 0),
        )

        # Check
        self.assertNotEqual(
            image_color_map[0][1], test_result_not_lenna_first_color_map
        )

    def test_image_detach_with_id_color_list(self):
        """
        `image_detach_with_id_color_list()` 메서드를 테스트합니다.
        """
        # Prerequisite
        image: np.ndarray = self.__get_sample_image()
        image_colors: List[Tuple[int, int, int]] = list(
            map(lambda v: v[0], image_info.get_all_colors(image, True))
        )
        image_color_map: List[
            Tuple[int, Tuple[int, int, int]]
        ] = image_color_transform.color_map_generate(image_colors, True)

        # Processing
        bin_num: int = 30
        mask_value: float = 1.0
        detached_image: np.ndarray = image_color_transform.image_detach_with_id_color_list(
            color_img=image,
            id_color_list=image_color_map[:bin_num],
            bin_num=bin_num,
            mask_value=mask_value,
        )

        # Result
        test_result_non_zero_counts: int = 32400
        test_result_unique_values: Tuple[float, float] = [0.0, mask_value]

        # Check
        self.assertEqual(np.count_nonzero(detached_image), test_result_non_zero_counts)
        self.assertEqual(np.unique(detached_image).tolist(), test_result_unique_values)

    def test_image_detach_with_id_color_probability_list(self):
        """
        `image_detach_with_id_color_probability_list()` 메서드를 테스트합니다.
        """
        # Prerequisite
        image: np.ndarray = self.__get_sample_image()
        image_colors: List[Tuple[int, int, int]] = list(
            map(lambda v: v[0], image_info.get_all_colors(image, True))
        )
        image_color_map: List[
            Tuple[int, Tuple[int, int, int]]
        ] = image_color_transform.color_map_generate(image_colors, True)

        # Processing
        bin_num: int = 30
        detached_image: np.ndarray = image_color_transform.image_detach_with_id_color_probability_list(
            color_img=image,
            id_color_list=image_color_map[:bin_num],
            bin_num=bin_num,
            resize_by_power_of_two=2,
        )

        # Result
        test_result_non_zero_counts: int = 2189
        test_result_unique_values: List[Tuple[float, int]] = [
            (0.0, 58561),
            (0.0625, 8),
            (0.25, 152),
            (0.4375, 8),
            (0.5, 76),
            (1.0, 1945),
        ]

        # Check
        self.assertEqual(np.count_nonzero(detached_image), test_result_non_zero_counts)
        self.assertEqual(
            np.unique(detached_image).tolist(),
            list(map(lambda v: v[0], test_result_unique_values)),
        )
        for value in test_result_unique_values:
            self.assertEqual(
                np.count_nonzero(np.equal(detached_image, value[0])), value[1]
            )

    def test_shrink(self):
        # Prerequisite
        a: np.ndarray = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        )

        # Processing
        shrinked_a = image_color_transform.shrink(a, 2, 2)
        shrinked_a2 = image_color_transform.shrink(a, 2, 1)

        # Result
        result_a: List[List[int]] = [[14, 22], [46, 54]]
        result_a2: List[List[int]] = [[36], [100]]

        # Check
        self.assertEqual(shrinked_a.tolist(), result_a)
        self.assertEqual(shrinked_a2.tolist(), result_a2)

    def test_shrink3D(self):
        # Prerequisite
        a: np.ndarray = np.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
                [[13, 14], [15, 16]],
            ]
        )

        # Processing
        shrinked_a = image_color_transform.shrink3D(a, 2, 1, 2)
        shrinked_a2 = image_color_transform.shrink3D(a, 2, 1, 1)

        # Result
        result_a: List[List[List[int]]] = [[[16, 20]], [[48, 52]]]
        result_a2: List[List[List[int]]] = [[[36]], [[100]]]

        # Check
        self.assertEqual(shrinked_a.tolist(), result_a)
        self.assertEqual(shrinked_a2.tolist(), result_a2)
