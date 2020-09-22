import math
import os
import time

import common_py
import cv2
import keras
import numpy as np
import tensorflow as tf
import toolz
from image_keras.batch_transform import generate_iterator_and_transform
from image_keras.flow_directory import FlowFromDirectory, ImagesFromDirectory
from image_keras.inout_generator import (
    BaseInOutGenerator,
    FlowManager,
    save_batch_transformed_img,
)
from image_keras.model_io import load_model
from image_keras.utils.image_transform import (
    gray_image_apply_clahe,
    img_to_ratio,
    ratio_to_img,
)
from keras.preprocessing.image import ImageDataGenerator

from models.semantic_segmentation.unet_l4.unet_l4 import UnetL4ModelHelper

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


if __name__ == "__main__":
    # 0. Prepare
    # ----------
    # predict_id: 사용한 모델, Predict 날짜
    # 0.1 ID ---------
    model_name: str = "unet_l4"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    predict_id: str = "_predict__model_{}__run_{}".format(model_name, run_id)

    # 0.2 Folder ---------
    # a) model, weights, result
    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    predict_result_folder: str = os.path.join(base_data_folder, predict_id)
    common_py.create_folder(predict_result_folder)

    # b) dataset folders
    predict_dataset_folder: str = os.path.join(base_data_folder, "ivan_filtered_test")
    # input - image
    image_folder: str = os.path.join(predict_dataset_folder, "image", "current")

    # 1. Model
    # --------
    # model -> load weights
    model_helper = UnetL4ModelHelper()
    # model (from python code)
    model = model_helper.get_model()
    # load weights
    weights_path: str = os.path.join(save_weights_folder, "unet010.hdf5")
    model.load_weights(weights_path)

    # 2. Dataset
    # ----------
    batch_size: int = 1

    # 2.1 Input ---------
    input_sizes = model_helper.model_descriptor.get_input_sizes()
    # a) image
    predict_img_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=image_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    predict_each_image_transform_function = gray_image_apply_clahe
    predict_each_transformed_image_save_function_optional = toolz.curry(
        save_batch_transformed_img
    )(predict_result_folder, "image_")
    predict_image_flow_manager: FlowManager = FlowManager(
        flow_from_directory=predict_img_flow,
        resize_to=input_sizes[0],
        image_transform_function=predict_each_image_transform_function,
        each_transformed_image_save_function_optional=predict_each_transformed_image_save_function_optional,
        transform_function_for_all=img_to_ratio,
    )

    # 2.2 Inout ---------
    in_out_generator = BaseInOutGenerator(input_flows=[predict_image_flow_manager])
    predict_generator = in_out_generator.get_generator()

    samples = in_out_generator.get_samples()
    filenames = in_out_generator.get_filenames()
    nb_samples = math.ceil(samples / batch_size)

    # 3. Predict
    # -------
    results = model.predict_generator(predict_generator, steps=nb_samples, verbose=1)

    # 4. Post Processing
    # ------------------
    for index, result in enumerate(results):
        name: str = os.path.basename(filenames[index])
        print("Post Processing for {}".format(name))
        full_path: str = os.path.join(predict_result_folder, name)
        result = ratio_to_img(result)
        cv2.imwrite(full_path, result)
