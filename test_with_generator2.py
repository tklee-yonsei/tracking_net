import math
import os
import time

import common_py
import cv2
import keras
import numpy as np
import tensorflow as tf
import toolz
from keras.preprocessing.image import ImageDataGenerator

from idl.batch_transform import generate_iterator_and_transform
from idl.descriptor.inout_generator import BaseInOutGenerator, FlowManager
from idl.flow_directory import FlowFromDirectory, ImagesFromDirectory
from idl.metrics import mean_iou
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
    model_path: str = os.path.join(save_models_folder, "unet_l4_000.json")
    weights_path: str = os.path.join(save_weights_folder, "unet010.hdf5")
    model: keras.models.Model = load_model(model_path, weights_path)

    # generator
    batch_size: int = 1
    image_folder: str = os.path.join(test_dataset_folder, "image", "current")
    label_folder: str = os.path.join(test_dataset_folder, "semantic_label")

    # image
    img_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=image_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    each_image_transform_function = (
        toolz.compose_left(
            lambda _img: np.array(_img, dtype=np.uint8),
            gray_image_apply_clahe,
            lambda _img: np.reshape(_img, (_img.shape[0], _img.shape[1], 1)),
        ),
        None,
    )
    each_transformed_image_save_function_optional = toolz.curry(
        save_batch_transformed_img
    )(test_result_folder, "image_")
    image_flow_manager: FlowManager = FlowManager(
        flow_from_directory=img_flow,
        image_transform_function=each_image_transform_function,
        each_transformed_image_save_function_optional=each_transformed_image_save_function_optional,
        transform_function_for_all=img_to_ratio,
    )

    # label
    label_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=label_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    each_transformed_label_save_function_optional = toolz.curry(
        save_batch_transformed_img
    )(test_result_folder, "label_")
    label_flow_manager: FlowManager = FlowManager(
        flow_from_directory=label_flow,
        each_transformed_image_save_function_optional=each_transformed_label_save_function_optional,
        transform_function_for_all=img_to_ratio,
    )

    # inout
    in_out_generator = BaseInOutGenerator(
        input_flows=[image_flow_manager], output_flows=[label_flow_manager]
    )
    test_generator = in_out_generator.get_generator()

    samples = in_out_generator.get_samples()
    filenames = in_out_generator.get_filenames()
    nb_samples = math.ceil(samples / batch_size)

    # test
    test_steps = samples // batch_size
    model.compile(
        optimizer=keras.optimizers.Adam(lr=1e-4),
        loss=keras.losses.binary_crossentropy,
        metrics=["accuracy", mean_iou],
    )

    test_loss, test_acc, test_mean_iou = model.evaluate_generator(
        test_generator, steps=test_steps, verbose=1, max_queue_size=1
    )

    print("test_loss: {}".format(test_loss))
    print("test_acc: {}".format(test_acc))
    print("test_mean_iou: {}".format(test_mean_iou))
