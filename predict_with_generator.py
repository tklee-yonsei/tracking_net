import math
import os

import cv2
import keras
import numpy as np
import tensorflow as tf
import toolz

from idl.batch_transform import generate_iterator_and_transform
from idl.flow_directory import FlowFromDirectory, ImagesFromDirectory
from idl.model_io import load_model
from utils.image_transform import gray_image_apply_clahe

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

    # 1
    # 이미지 크기(채널)가 변하지 않을 경우, 이 방법을 권장.
    # 전처리 중간 결과 저장에서, 멀티쓰레드 처리에 의해 섞이는 순서를 그나마 컨트롤 할 수 있다.
    # img_flow: FlowFromDirectory = ImagesFromDirectory(
    #     dataset_directory=image_folder,
    #     batch_size=batch_size,
    #     color_mode="grayscale",
    #     shuffle=False,
    #     save_to_dir=".",
    #     save_prefix="pre_processed_img_",
    # )
    # idg = ImageDataGenerator(
    #     preprocessing_function=toolz.compose_left(
    #         lambda _img: np.array(_img, dtype=np.uint8),
    #         gray_image_apply_clahe,
    #         lambda _img: np.reshape(_img, (_img.shape[0], _img.shape[1], 1)),
    #     ),
    #     fill_mode="nearest",
    # )
    # image_generator = img_flow.get_iterator(generator=idg)
    # image_transformed_generator = generate_iterator_and_transform(
    #     image_generator=image_generator, transform_function_for_all=img_normalization,
    # )

    # 2
    # 각 이미지 처리 함수에서, 이미지 크기(채널)가 변하는 경우, 이 방법을 권장.
    # 중간 결과의 순서는 보장되지 않는다.
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
        each_transformed_image_save_function_optional=save_batch_transformed_img,
        transform_function_for_all=img_to_ratio,
    )

    filenames = image_generator.filenames
    nb_samples = math.ceil(image_generator.samples / batch_size)

    prediction_generator = image_transformed_generator

    # predict
    results = model.predict_generator(prediction_generator, steps=nb_samples, verbose=1)

    # 후처리 및 저장
    for index, result in enumerate(results):
        name: str = os.path.basename(filenames[index])
        print("Post Processing for {}".format(name))
        result = ratio_to_img(result)
        cv2.imwrite(name, result)
