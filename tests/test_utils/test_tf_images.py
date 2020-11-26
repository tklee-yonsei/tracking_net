import os
from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_array_ops
from utils import tf_images


class TestTFImage(TestCase):
    current_path: str = os.path.dirname(os.path.abspath(__file__))
    working_path: str = os.getcwd()
    test_path: str = os.path.join(working_path, "tests")
    test_resource_folder_name: str = "test_resources"

    def test_decode_image(self):
        # Expected Result
        sample_image_shape = [180, 180, 3]
        result = tf.constant(sample_image_shape)

        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        # 2) Calculate shape
        self.assertTrue(tf.math.reduce_all(tf.math.equal(tf.shape(tf_image), result)))

    def test_tf_img_to_minmax(self):
        # Expected Result
        threshold = 127
        min_maxed_unique_values = (255, 0)
        number_of_black_color = 31760

        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        grayscale_tf_image = tf_images.decode_png(test_image_fullpath, 1)

        # 2) Calculate min max
        min_maxed_grayscale_tf_image = tf_images.tf_img_to_minmax(
            grayscale_tf_image, threshold, (0, 255)
        )

        # check unique
        reshaped_min_maxed_grayscale_tf_image = tf.reshape(
            min_maxed_grayscale_tf_image, (-1, 1)
        )
        unique_min_max_values = gen_array_ops.unique_v2(
            reshaped_min_maxed_grayscale_tf_image, axis=[-2]
        )[0]

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(
                    unique_min_max_values,
                    tf.cast(
                        tf.expand_dims(tf.constant(min_maxed_unique_values), axis=-1),
                        tf.float32,
                    ),
                )
            )
        )
        self.assertTrue(
            tf.math.equal(
                tf.math.count_nonzero(min_maxed_grayscale_tf_image),
                number_of_black_color,
            )
        )

    def test_tf_get_all_colors(self):
        # Expected Result
        sample_image_colors = [
            [245.0, 245.0, 245.0],
            [71.0, 71.0, 71.0],
            [0.0, 0.0, 0.0],
            [255.0, 145.0, 77.0],
            [72.0, 72.0, 72.0],
        ]
        result = tf.constant(sample_image_colors)

        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        # 2) Calculate colors in image
        tf_image_colors = tf_images.tf_get_all_colors(tf_image)
        print(tf_image_colors)
        self.assertTrue(tf.math.reduce_all(tf.math.equal(tf_image_colors, result)))

    def test_tf_generate_color_map(self):
        # 1) Get sample image as tf
        test_file_name: str = "sample.png"
        test_image_fullpath: str = os.path.join(
            self.test_path, self.test_resource_folder_name, test_file_name
        )
        tf_image = tf_images.decode_png(test_image_fullpath, 3)

        a = tf_images.tf_generate_color_map(tf_image)
        print(a)

    def test_tf_shrink3D(self):
        # Expected Result
        result1 = tf.constant([[[16, 20]], [[48, 52]]], dtype=tf.int64)
        result2 = tf.constant([[[36]], [[100]]], dtype=tf.int64)

        a = tf.constant(
            np.array(
                [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                    [[13, 14], [15, 16]],
                ]
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(tf_images.tf_shrink3D(a, 2, 1, 2), result1)
            )
        )
        self.assertTrue(
            tf.math.reduce_all(
                tf.math.equal(tf_images.tf_shrink3D(a, 2, 1, 1), result2)
            )
        )

    def test_tf_extract_patches(self):
        sample_image = tf.constant(np.random.randint(10, size=(1, 5, 5, 1)))

        self.assertTrue(
            tf.math.reduce_all(
                tf.image.extract_patches(
                    sample_image,
                    sizes=[1, 3, 3, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                )
                == tf_images.tf_extract_patches(sample_image, 3)
            )
        )

        self.assertTrue(
            tf.math.reduce_all(
                tf.image.extract_patches(
                    sample_image,
                    sizes=[1, 5, 5, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                )
                == tf_images.tf_extract_patches(sample_image, 5)
            )
        )

        sample_image2 = tf.constant(np.random.randint(10, size=(1, 32, 32, 1)))
        self.assertTrue(
            tf.math.reduce_all(
                tf.image.extract_patches(
                    sample_image2,
                    sizes=[1, 5, 5, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                )
                == tf_images.tf_extract_patches(sample_image2, 5)
            )
        )

        sample_image3 = tf.constant(np.random.randint(10, size=(2, 32, 32, 30)))
        self.assertTrue(
            tf.math.reduce_all(
                tf.image.extract_patches(
                    sample_image3,
                    sizes=[1, 5, 5, 1],
                    strides=[1, 1, 1, 1],
                    rates=[1, 1, 1, 1],
                    padding="SAME",
                )
                == tf_images.tf_extract_patches(sample_image3, 5)
            )
        )
