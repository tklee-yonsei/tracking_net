import math
import os
import sys

sys.path.append(os.getcwd())

import time
from typing import Callable, Generator, List, Optional, Tuple

import common_py
import numpy as np
import toolz
from image_keras.flow_directory import FlowFromDirectory, ImagesFromDirectory
from image_keras.inout_generator import (
    BaseInOutGenerator,
    FlowManager,
    save_batch_transformed_img,
)
from image_keras.utils.image_transform import img_to_ratio
from models.semantic_segmentation.unet_l4.config import (
    UnetL4ModelHelper,
    input_image_preprocessing_function,
    output_label_preprocessing_function,
)

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
    base_dataset_folder: str = os.path.join("dataset")
    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    test_result_folder: str = os.path.join(base_data_folder, test_id)
    common_py.create_folder(test_result_folder)

    # b) dataset folders
    test_dataset_folder: str = os.path.join(
        base_dataset_folder, "test_original_20_edge10"
    )
    # input - image
    test_image_folder: str = os.path.join(test_dataset_folder, "image")
    # output - label
    test_label_folder: str = os.path.join(test_dataset_folder, "label")

    # 1. Model
    # --------
    model_helper = UnetL4ModelHelper()
    model = model_helper.get_model()
    model = model_helper.compile_model(model)
    weights_path: str = os.path.join(save_weights_folder, "unet010.hdf5")
    model.load_weights(weights_path)

    # 2. Dataset
    # ----------
    batch_size: int = 8

    # 2.1 Input ---------
    input_sizes = model_helper.model_descriptor.get_input_sizes()

    # a) image
    def __input_image_flow(
        dataset_directory: str,
        batch_size: int,
        shuffle: bool,
        preprocessing_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        save_folder_and_prefix: Optional[Tuple[str, str]] = None,
        seed: int = 42,
    ) -> FlowManager:
        _each_transformed_image_save_function_optional = None
        if save_folder_and_prefix is not None:
            _each_transformed_image_save_function_optional = toolz.curry(
                save_batch_transformed_img
            )(save_folder_and_prefix[0], save_folder_and_prefix[1])
        _img_flow: FlowFromDirectory = ImagesFromDirectory(
            dataset_directory=dataset_directory,
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle=shuffle,
            seed=seed,
        )
        _image_flow_manager: FlowManager = FlowManager(
            flow_from_directory=_img_flow,
            resize_to=input_sizes[0],
            image_transform_function=preprocessing_function,
            each_transformed_image_save_function_optional=_each_transformed_image_save_function_optional,
            transform_function_for_all=img_to_ratio,
        )
        return _image_flow_manager

    test_image_flow_manager = __input_image_flow(
        dataset_directory=test_image_folder,
        batch_size=batch_size,
        preprocessing_function=input_image_preprocessing_function,
        # save_folder_and_prefix=(training_result_folder, "training_image_"),
        shuffle=True,
    )

    # 2.2 Output ---------
    output_sizes = model_helper.model_descriptor.get_output_sizes()

    # a) label
    def __output_label_flow(
        dataset_directory: str,
        batch_size: int,
        shuffle: bool,
        preprocessing_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        save_folder_and_prefix: Optional[Tuple[str, str]] = None,
        seed: int = 42,
    ) -> FlowManager:
        _each_transformed_label_save_function_optional = None
        if save_folder_and_prefix is not None:
            _each_transformed_label_save_function_optional = toolz.curry(
                save_batch_transformed_img
            )(save_folder_and_prefix[0], save_folder_and_prefix[1])
        _label_flow: FlowFromDirectory = ImagesFromDirectory(
            dataset_directory=dataset_directory,
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle=shuffle,
            seed=seed,
        )
        _label_flow_manager: FlowManager = FlowManager(
            flow_from_directory=_label_flow,
            resize_to=output_sizes[0],
            image_transform_function=preprocessing_function,
            each_transformed_image_save_function_optional=_each_transformed_label_save_function_optional,
            transform_function_for_all=img_to_ratio,
        )
        return _label_flow_manager

    test_label_flow_manager = __output_label_flow(
        dataset_directory=test_label_folder,
        batch_size=batch_size,
        preprocessing_function=output_label_preprocessing_function,
        # save_folder_and_prefix=(training_result_folder, "training_label_"),
        shuffle=True,
    )

    # 2.3 Inout ---------
    def __inout_generator_infos(
        input_flows: List[FlowManager], output_flows: List[FlowManager]
    ) -> Tuple[Generator, int, int, List[str]]:
        _in_out_generator = BaseInOutGenerator(
            input_flows=input_flows, output_flows=output_flows,
        )

        _generator = _in_out_generator.get_generator()
        _samples = _in_out_generator.get_samples()
        _nb_samples = math.ceil(_samples / batch_size)

        return _generator, _samples, _nb_samples

    (test_generator, test_samples, test_nb_samples) = __inout_generator_infos(
        input_flows=[test_image_flow_manager], output_flows=[test_label_flow_manager],
    )

    # 3. Test
    # -------
    test_steps = test_samples // batch_size
    test_loss, test_acc, test_mean_iou = model.evaluate_generator(
        test_generator, steps=test_steps, verbose=1, max_queue_size=1
    )

    # print(
    #     "loss : {}".format(
    #         dict(map(lambda k: (k[0], k[1].name), model.compiled_loss._losses.items()))
    #     )
    # )
    print("loss : {}".format(list(map(lambda k: k.name, model.compiled_loss._losses))))
    print("loss weights : {}".format(model.compiled_loss._loss_weights))
    print(
        "metrics : {}".format(
            list(
                map(
                    lambda el: list(map(lambda el2: el2.name, el)),
                    model.compiled_metrics._metrics,
                )
            )
        )
    )

    print("test_loss: {}".format(test_loss))
    print("test_acc: {}".format(test_acc))
    print("test_mean_iou: {}".format(test_mean_iou))
