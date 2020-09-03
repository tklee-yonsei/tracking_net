from idl.flow_directory import FlowFromDirectory, ImagesFromDirectory
import os
from typing import Callable, Tuple
from unittest import TestCase

import numpy as np

from idl import batch_transform


class TestBatchTransform(TestCase):
    current_path: str = os.path.dirname(os.path.abspath(__file__))
    working_path: str = os.getcwd()
    test_path: str = os.path.join(working_path, "tests")
    test_resource_folder_name: str = "test_resources"
    full_test_resource_folder: str = os.path.join(test_path, test_resource_folder_name)

    def test_transform_for_batch(self):
        """
        `transform_for_batch()` 메서드를 테스트합니다.

        임의의 한 배치 이미지를, 메서드를 활용한 변환한 후, 비교 테스트합니다.
        """
        # Prerequisite
        batch_img: np.ndarray = np.arange(0, 96).reshape(2, 4, 4, 3)

        # Processing
        batch_transform_function: Callable[[np.ndarray], np.ndarray] = lambda v: v + 1
        batch_transform_function_output_shape: Tuple[int, int, int] = (2, 4, 4, 3)
        transformed: np.ndarray = batch_transform.transform_for_batch(
            batch_img, batch_transform_function, batch_transform_function_output_shape
        )

        # Result
        result_transform: np.ndarray = np.arange(1, 97).reshape(2, 4, 4, 3)

        # Check
        self.assertTrue(np.array_equal(transformed, result_transform))

    def test_example_generate_iterator_and_transform(self):
        image_from_directory: FlowFromDirectory = ImagesFromDirectory(
            self.full_test_resource_folder, 1
        )
        # Prerequisite

        # Processing
        iterator = batch_transform.generate_iterator_and_transform(image_from_directory)

        # Result

        # Check
        self.assertTrue(True)
