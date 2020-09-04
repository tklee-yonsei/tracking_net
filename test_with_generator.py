import math
import os
from typing import Callable

import cv2
import keras
import numpy as np
import toolz
from keras.preprocessing.image import ImageDataGenerator

from idl.batch_transform import generate_iterator_and_transform
from idl.flow_directory import FlowFromDirectory, ImagesFromDirectory
from idl.model_io import load_model
from utils.image_transform import gray_image_apply_clahe


def img_to_ratio(img: np.ndarray) -> np.ndarray:
    return img / 255.0


def ratio_to_img(ratio_img: np.ndarray) -> np.ndarray:
    return ratio_img * 255


def ratio_threshold(ratio_img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    result_ratio_img = ratio_img.copy()
    result_ratio_img[result_ratio_img > threshold] = 1
    result_ratio_img[result_ratio_img <= threshold] = 0
    return result_ratio_img


def save_batch_transformed_img(
    index_num: int, batch_num: int, transformed_batch_img: np.ndarray
) -> None:
    # 이름
    img_path = os.path.join(".")
    img_name = "img_transformed_{:04d}_{:02d}.png".format(index_num, batch_num)
    img_fullpath = os.path.join(img_path, img_name)

    # 저장
    cv2.imwrite(img_fullpath, transformed_batch_img)


if __name__ == "__main__":
    # 사용한 모델, 사용한 트레이닝 Prediction 날짜
    prediction_id: str = "_testest"

    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")

    prediction_dataset_folder: str = os.path.join(
        base_data_folder, "ivan_filtered_test"
    )

    # model
    model_path: str = os.path.join(save_models_folder, "unet_l4.json")
    weights_path: str = os.path.join(save_weights_folder, "unet010.hdf5")
    model: keras.models.Model = load_model(model_path, weights_path)

    # generator
    batch_size: int = 1
    image_folder: str = os.path.join(prediction_dataset_folder, "image", "current")
    label_folder: str = os.path.join(prediction_dataset_folder, "semantic_label")

    img_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=image_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    image_generator = img_flow.get_iterator()
    image_transformed_generator = generate_iterator_and_transform(
        image_generator=image_generator,
        each_image_transform_function=(
            toolz.compose_left(
                lambda _img: np.array(_img, dtype=np.uint8),
                gray_image_apply_clahe,
                lambda _img: np.reshape(_img, (_img.shape[0], _img.shape[1], 1)),
            ),
            None,
        ),
        transform_function_for_all=img_to_ratio,
    )

    label_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=label_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    label_generator = label_flow.get_iterator()
    label_transformed_generator = generate_iterator_and_transform(
        image_generator=label_generator, transform_function_for_all=img_to_ratio,
    )

    filenames = image_generator.filenames
    nb_samples = math.ceil(image_generator.samples / batch_size)

    input_generator = map(list, zip(image_transformed_generator))
    output_generator = map(list, zip(label_transformed_generator))
    test_generator = zip(input_generator, output_generator)

    # test
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-4),
        loss=keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )
    test_loss, test_acc = model.evaluate_generator(
        test_generator, steps=nb_samples, verbose=1
    )

    print("test_loss: {}".format(test_loss))
    print("test_acc: {}".format(test_acc))
