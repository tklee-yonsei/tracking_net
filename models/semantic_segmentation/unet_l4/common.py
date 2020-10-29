import math
import os
from random import Random
from typing import Callable, List, Optional

import cv2
import numpy as np
import tensorflow as tf


class UNetL4Sequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        image_file_names: List[str],
        main_image_folder_name: str,
        output_label_folder_name: str,
        main_image_preprocessing_function: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = None,
        output_label_preprocessing_function: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.image_file_names = image_file_names
        self.main_image_folder_name = main_image_folder_name
        self.output_label_folder_name = output_label_folder_name
        self.main_image_preprocessing_function = main_image_preprocessing_function
        self.output_label_preprocessing_function = output_label_preprocessing_function
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        if shuffle is True:
            Random(seed).shuffle(self.image_file_names)

    def __len__(self):
        return math.ceil(len(self.image_file_names) / self.batch_size)

    def __getitem__(self, idx):
        batch_image_file_names = self.image_file_names[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        batch_main_images = []
        batch_output_labels = []
        for image_file_name in batch_image_file_names:
            main_img = cv2.imread(
                os.path.join(self.main_image_folder_name, image_file_name),
                cv2.IMREAD_GRAYSCALE,
            )
            if self.main_image_preprocessing_function is not None:
                main_img = self.main_image_preprocessing_function(main_img)
            main_img = main_img.reshape((main_img.shape[0], main_img.shape[1], 1))
            batch_main_images.append(main_img)

            output_label = cv2.imread(
                os.path.join(self.output_label_folder_name, image_file_name),
                cv2.IMREAD_GRAYSCALE,
            )
            output_label = self.output_label_preprocessing_function(output_label)
            output_label = output_label.reshape(
                (output_label.shape[0], output_label.shape[1], 1)
            )
            batch_output_labels.append(output_label)

        X = [np.array(batch_main_images)]
        Y = [np.array(batch_output_labels)]

        return (X, Y)
