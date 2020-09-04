import os
from typing import Callable, List, Optional, Tuple

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skimage.io as io
import skimage.transform as trans

from seg_models.pre_process import gray_image_to_ratio, gray_image_to_threshold_ratio


def adjust_data(img, label):
    img = gray_image_to_ratio(img)
    label = gray_image_to_threshold_ratio(label)
    return img, label


def image_label_set_generator(
    batch_size: int,
    directory: str,
    image_folder: str,
    label_folder: str,
    image_data_generator: ImageDataGenerator,
    label_data_generator: ImageDataGenerator,
    image_color_mode: str = "grayscale",
    label_color_mode: str = "grayscale",
    image_save_prefix: str = "image",
    label_save_prefix: str = "label",
    save_to_dir=None,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 1,
):
    """
    Generate image and label at the same time.

    Use the same seed for image_data_generator and label_data_generator to ensure the transformation for image and
    label is the same.
    If you want to visualize the results of generator, set save_to_dir = "your path"

    Args:
        batch_size (int): Size of the batches of data
        directory (str): Path to the target directory. It should contain one subdirectory per class.
        image_folder (str): Image data directory
        label_folder (str):
        image_data_generator (ImageDataGenerator):
        label_data_generator (ImageDataGenerator):
        image_color_mode:
        label_color_mode:
        image_save_prefix:
        label_save_prefix:
        save_to_dir:
        target_size:
        seed:

    Returns:
        Generator for image, label tuple
    """
    image_generator = image_data_generator.flow_from_directory(
        directory,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed,
    )
    label_generator = label_data_generator.flow_from_directory(
        directory,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed,
    )
    train_generators = zip(image_generator, label_generator)
    for img, label in train_generators:
        img, label = adjust_data(img, label)
        yield img, label


def img_read_preprocessing(
    path_file: str,
    optional_pre_processing_function: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
    target_size=(256, 256),
    as_gray=True,
):
    img = io.imread(path_file, as_gray=as_gray)

    img = np.array(img, dtype=np.uint8)
    img = (
        optional_pre_processing_function(img)
        if optional_pre_processing_function is not None
        else img
    )
    img = gray_image_to_ratio(img)

    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)
    return img


def predict_generator(
    path: str,
    files: List[str],
    optional_pre_processing_function: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
    target_size=(256, 256),
    as_gray=True,
):
    for file in files:
        import cv2

        img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        # img = io.imread(os.path.join(path, file), as_gray=as_gray)

        img = np.array(img, dtype=np.uint8)
        if optional_pre_processing_function:
            img = optional_pre_processing_function(img)
        img = gray_image_to_ratio(img)

        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img


def save_result_with_name(
    save_path: str,
    numpy_images: List[np.ndarray],
    name_list: List[str],
    image_postprocessing_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    for index, numpy_image in enumerate(numpy_images):
        o = (255 * numpy_image[:, :, 0]).astype(np.uint8)
        if image_postprocessing_function:
            o = image_postprocessing_function(o)
        io.imsave(os.path.join(save_path, name_list[index]), o, check_contrast=False)
