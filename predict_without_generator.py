import os
import time

import common_py
import cv2
import keras
import numpy as np
import tensorflow as tf
import toolz

from idl.model_io import load_model
from utils.image_transform import gray_image_apply_clahe, img_to_ratio, ratio_to_img

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
    index_num: int,
    batch_num: int,
    transformed_batch_img: np.ndarray,
) -> None:
    # 이름
    img_name = "img_transformed_{:04d}_{:02d}.png".format(index_num, batch_num)
    img_fullpath = os.path.join(target_folder, img_name)

    # 저장
    cv2.imwrite(img_fullpath, transformed_batch_img)


if __name__ == "__main__":
    # prediction_id: 사용한 모델, Prediction 날짜
    model_name: str = "unet_l4"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    prediction_id: str = "_predict__model_{}__run_{}".format(model_name, run_id)

    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")

    prediction_dataset_folder: str = os.path.join(
        base_data_folder, "ivan_filtered_test"
    )
    prediction_result_folder: str = os.path.join(base_data_folder, prediction_id)
    common_py.create_folder(prediction_result_folder)

    # model
    model_path: str = os.path.join(save_models_folder, "unet_l4_000.json")
    weights_path: str = os.path.join(save_weights_folder, "unet010.hdf5")
    model: keras.models.Model = load_model(model_path, weights_path)

    # generator
    batch_size: int = 1
    image_folder: str = os.path.join(prediction_dataset_folder, "image", "current")

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

    input_generator = map(list, zip(image_transformed_generator))

    # predict
    results = model.predict_generator(input_generator, steps=5, verbose=1)

    # 후처리 및 저장
    for index, result in enumerate(results):
        name: str = os.path.basename(image_files[index])
        print("Post Processing for {}".format(name))
        full_path: str = os.path.join(prediction_result_folder, name)
        result = ratio_to_img(result)
        cv2.imwrite(full_path, result)
