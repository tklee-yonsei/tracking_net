import math
import os
import time

import common_py
import cv2
import keras
import numpy as np
import tensorflow as tf
import toolz

from idl.model_io import load_model
from utils.image_transform import gray_image_apply_clahe, img_to_ratio

# GPU Setting
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    try:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
        print(e)


def save_batch_transformed_img(
    target_folder: str,
    prefix: str,
    index_num: int,
    batch_num: int,
    transformed_batch_img: np.ndarray,
) -> None:
    # 이름
    img_name = "{}img_transformed_{:04d}_{:02d}.png".format(
        prefix, index_num, batch_num
    )
    img_fullpath = os.path.join(target_folder, img_name)

    # 저장
    cv2.imwrite(img_fullpath, transformed_batch_img)


if __name__ == "__main__":
    # test_id: 사용한 모델, Test 날짜
    model_name: str = "unet_l4"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    test_id: str = "_test__model_{}__run_{}".format(model_name, run_id)

    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")

    test_dataset_folder: str = os.path.join(base_data_folder, "ivan_filtered_test")
    test_result_folder: str = os.path.join(base_data_folder, test_id)
    common_py.create_folder(test_result_folder)

    # model
    model_path: str = os.path.join(save_models_folder, "unet_l4.json")
    weights_path: str = os.path.join(save_weights_folder, "unet010.hdf5")
    model: keras.models.Model = load_model(model_path, weights_path)

    # generator
    batch_size: int = 1
    image_folder: str = os.path.join(test_dataset_folder, "image", "current")
    label_folder: str = os.path.join(test_dataset_folder, "semantic_label")

    image_files = common_py.files_in_folder(image_folder)
    image_files = sorted(image_files)

    def xx(path, image_files):
        for image_file in image_files:
            img = cv2.imread(os.path.join(path, image_file), cv2.IMREAD_GRAYSCALE)
            pre_processed_img = toolz.compose_left(
                lambda _img: np.array(_img, dtype=np.uint8),
                gray_image_apply_clahe,
                img_to_ratio,
                lambda _img: np.reshape(_img, (1, _img.shape[0], _img.shape[1], 1)),
            )(img)
            yield pre_processed_img

    image_transformed_generator = xx(image_folder, image_files)

    label_files = common_py.files_in_folder(label_folder)
    label_files = sorted(label_files)

    def xx2(path, __label_files):
        for _label_file in __label_files:
            img2 = cv2.imread(os.path.join(path, _label_file), cv2.IMREAD_GRAYSCALE)
            pre_processed_img2 = toolz.compose_left(
                lambda _img: np.array(_img, dtype=np.uint8),
                img_to_ratio,
                lambda _img: np.reshape(_img, (1, _img.shape[0], _img.shape[1], 1)),
            )(img2)
            yield pre_processed_img2

    label_transformed_generator = xx2(label_folder, label_files)

    input_generator = map(list, zip(image_transformed_generator))
    output_generator = map(list, zip(label_transformed_generator))
    test_generator = zip(input_generator, output_generator)

    # test
    test_steps = len(label_files)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-4),
        loss=keras.losses.binary_crossentropy,
        metrics=["accuracy"],
    )

    test_loss, test_acc = model.evaluate_generator(
        test_generator, steps=test_steps, verbose=1, max_queue_size=1
    )

    print("test_loss: {}".format(test_loss))
    print("test_acc: {}".format(test_acc))
