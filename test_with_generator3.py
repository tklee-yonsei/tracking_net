import math
import os
import time

import common_py
import tensorflow as tf
import toolz
from image_keras.flow_directory import FlowFromDirectory, ImagesFromDirectory
from image_keras.inout_generator import (
    BaseInOutGenerator,
    FlowManager,
    save_batch_transformed_img,
)
from image_keras.model_io import load_model, save_model
from image_keras.utils.image_transform import (
    gray_image_apply_clahe,
    img_to_minmax,
    img_to_ratio,
)

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
    # test_id: 사용한 모델, Test 날짜
    # 0.1 ID ---------
    model_name: str = "unet_l4"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    test_id: str = "_test__model_{}__run_{}".format(model_name, run_id)

    # 0.2 Folder ---------
    # a) model, weights, result
    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    test_result_folder: str = os.path.join(base_data_folder, test_id)
    common_py.create_folder(test_result_folder)

    # a) dataset folders
    test_dataset_folder: str = os.path.join(base_data_folder, "ivan_filtered_test")
    # input - image
    image_folder: str = os.path.join(test_dataset_folder, "image", "current")
    # output - label
    label_folder: str = os.path.join(test_dataset_folder, "semantic_label")

    # 1. Model
    # --------
    # model -> compile -> load weights
    model_helper = UnetL4ModelHelper()
    # model (from python code)
    # model = model_helper.get_model()

    # compile
    # model = model_helper.compile_model(model)

    # save model to json
    # model_path: str = os.path.join(save_models_folder, "unet_l4_001.json")
    # save_model(model_path, model)

    # model (from json)
    model_path: str = os.path.join(save_models_folder, "unet_l4_001.json")
    weights_path: str = os.path.join(save_weights_folder, "unet010.hdf5")
    model = load_model(model_path)

    # compile
    model = model_helper.compile_model(model)

    # load weights
    weights_path: str = os.path.join(save_weights_folder, "unet010.hdf5")
    model.load_weights(weights_path)

    # 2. Dataset
    # ----------
    batch_size: int = 1

    # 2.1 Input ---------
    input_sizes = model_helper.model_descriptor.get_input_sizes()
    # a) image
    test_img_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=image_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    test_each_image_transform_function = gray_image_apply_clahe
    test_each_transformed_image_save_function_optional = toolz.curry(
        save_batch_transformed_img
    )(test_result_folder, "image_")
    test_image_flow_manager: FlowManager = FlowManager(
        flow_from_directory=test_img_flow,
        resize_to=input_sizes[0],
        image_transform_function=test_each_image_transform_function,
        each_transformed_image_save_function_optional=test_each_transformed_image_save_function_optional,
        transform_function_for_all=img_to_ratio,
    )

    # 2.2 Output ---------
    output_sizes = model_helper.model_descriptor.get_output_sizes()
    # a) label
    test_label_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=label_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    test_each_label_transform_function = toolz.compose_left(
        lambda _img: img_to_minmax(_img, 127, (0, 255)),
    )
    test_each_transformed_label_save_function_optional = toolz.curry(
        save_batch_transformed_img
    )(test_result_folder, "label_")
    test_label_flow_manager: FlowManager = FlowManager(
        flow_from_directory=test_label_flow,
        resize_to=output_sizes[0],
        image_transform_function=test_each_label_transform_function,
        each_transformed_image_save_function_optional=test_each_transformed_label_save_function_optional,
        transform_function_for_all=img_to_ratio,
    )

    # 2.3 Inout ---------
    in_out_generator = BaseInOutGenerator(
        input_flows=[test_image_flow_manager], output_flows=[test_label_flow_manager]
    )
    test_generator = in_out_generator.get_generator()

    samples = in_out_generator.get_samples()
    filenames = in_out_generator.get_filenames()
    nb_samples = math.ceil(samples / batch_size)

    # 3. Test
    # -------
    test_steps = samples // batch_size
    test_loss, test_acc, test_mean_iou = model.evaluate_generator(
        test_generator, steps=test_steps, verbose=1, max_queue_size=1
    )

    print(
        "loss : {}".format(
            dict(map(lambda kv: (kv[0], kv[1].name), model.loss.items()))
        )
    )
    print("loss weights : {}".format(model.loss_weights))
    print("metrics : {}".format(list(map(lambda el: el.name, model.metrics))))

    print("test_loss: {}".format(test_loss))
    print("test_acc: {}".format(test_acc))
    print("test_mean_iou: {}".format(test_mean_iou))
