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
    DistFlowManager,
    DistInOutGenerator,
    Distributor,
    save_batch_transformed_img,
)
from image_keras.utils.image_transform import img_to_ratio
from models.color_tracking.model_006 import (
    Model006ModelHelper,
    input_main_image_preprocessing_function,
    input_ref1_label_preprocessing_function,
    input_ref2_label_preprocessing_function,
    input_ref3_label_preprocessing_function,
    input_ref_image_preprocessing_function,
)

if __name__ == "__main__":
    # 0. Prepare
    # ----------

    # predict_id: 사용한 모델, Predict 날짜
    # 0.1 ID ---------
    model_name: str = "model_006"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    predict_id: str = "_predict__model_{}__run_{}".format(model_name, run_id)

    # 0.2 Folder ---------

    # a) model, weights, result
    base_dataset_folder: str = os.path.join("dataset")
    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    predict_result_folder: str = os.path.join(base_data_folder, predict_id)
    common_py.create_folder(predict_result_folder)

    # b) dataset folders
    predict_dataset_folder: str = os.path.join(
        base_dataset_folder, "ivan_filtered_test"
    )
    # input - main image
    predict_main_image_folder: str = os.path.join(
        predict_dataset_folder, "image", "current"
    )
    # input - ref image
    predict_ref_image_folder: str = os.path.join(
        predict_dataset_folder, "image", "prev"
    )
    # input - ref result label
    predict_ref_result_label_folder: str = os.path.join(
        predict_dataset_folder, "prev_result"
    )
    # output - main label
    predict_output_main_label_folder: str = os.path.join(
        predict_dataset_folder, "label"
    )

    # 1. Model
    # --------
    # model -> compile
    # a) model (from python code)
    from models.semantic_segmentation.unet_l4.config import UnetL4ModelHelper

    unet_model_helper = UnetL4ModelHelper()
    unet_model = unet_model_helper.get_model()
    model_helper = Model006ModelHelper(pre_trained_unet_l4_model=unet_model)
    model = model_helper.get_model()

    # b) compile
    model = model_helper.compile_model(model)

    # c) load weights
    weights_path: str = os.path.join(
        save_weights_folder,
        # "training__model_model_006__run_20200925-091431.epoch_02-val_loss_0.014-val_accuracy_0.952.hdf5",
        "training__model_model_006__run_20200928-150618.epoch_01-val_loss_0.084-val_acc_0.960.hdf5",
    )
    model.load_weights(weights_path)

    # 2. Dataset
    # ----------
    predict_batch_size: int = 1

    # 2.1 Input ---------
    input_sizes = model_helper.model_descriptor.get_input_sizes()

    # a) main image
    def __input_main_image_flow(
        dataset_directory: str,
        batch_size: int,
        shuffle: bool,
        preprocessing_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        save_folder_and_prefix: Optional[Tuple[str, str]] = None,
        seed: int = 42,
    ) -> DistFlowManager:
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
        _distributor: Distributor = Distributor(
            resize_to=input_sizes[0],
            image_transform_function=preprocessing_function,
            each_transformed_image_save_function_optional=_each_transformed_image_save_function_optional,
            transform_function_for_all=img_to_ratio,
        )
        _image_flow_manager: DistFlowManager = DistFlowManager(
            flow_from_directory=_img_flow, distributors=[_distributor],
        )
        return _image_flow_manager

    predict_main_image_flow_manager = __input_main_image_flow(
        dataset_directory=predict_main_image_folder,
        batch_size=predict_batch_size,
        preprocessing_function=input_main_image_preprocessing_function,
        # save_folder_and_prefix=(training_result_folder, "training_main_image_"),
        shuffle=False,
    )

    # b) ref image
    def __input_ref_image_flow(
        dataset_directory: str,
        batch_size: int,
        shuffle: bool,
        preprocessing_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        save_folder_and_prefix: Optional[Tuple[str, str]] = None,
        seed: int = 42,
    ) -> DistFlowManager:
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
        _distributor: Distributor = Distributor(
            resize_to=input_sizes[0],
            image_transform_function=preprocessing_function,
            each_transformed_image_save_function_optional=_each_transformed_image_save_function_optional,
            transform_function_for_all=img_to_ratio,
        )
        _image_flow_manager: DistFlowManager = DistFlowManager(
            flow_from_directory=_img_flow, distributors=[_distributor],
        )
        return _image_flow_manager

    predict_ref_image_flow_manager = __input_ref_image_flow(
        dataset_directory=predict_ref_image_folder,
        batch_size=predict_batch_size,
        preprocessing_function=input_ref_image_preprocessing_function,
        # save_folder_and_prefix=(training_result_folder, "training_ref_image_"),
        shuffle=False,
    )

    # c) ref result label
    def __input_result_label_flow(
        dataset_directory: str,
        batch_size: int,
        shuffle: bool,
        distributors: List[Distributor],
        seed: int = 42,
    ) -> DistFlowManager:
        _img_flow: FlowFromDirectory = ImagesFromDirectory(
            dataset_directory=dataset_directory,
            batch_size=batch_size,
            color_mode="rgb",
            shuffle=shuffle,
            seed=seed,
        )
        _image_flow_manager: DistFlowManager = DistFlowManager(
            flow_from_directory=_img_flow, distributors=distributors,
        )
        return _image_flow_manager

    def save_batch_transformed_img2(
        target_folder: str,
        prefix: str,
        index_num: int,
        batch_num: int,
        image: np.ndarray,
    ) -> None:
        """
        배치 변환된 이미지를 저장합니다.

        Parameters
        ----------
        target_folder : str
            타겟 폴더
        prefix : str
            파일의 맨 앞에 붙을 prefix
        index_num : int
            파일의 인덱스 번호
        batch_num : int
            파일의 배치 번호
        image : np.ndarray
            이미지
        """
        import cv2

        for i in range(image.shape[2]):
            img_name = "{}img_transformed_{:04d}_{:02d}_{:02d}.png".format(
                prefix, index_num, batch_num, i
            )
            img_fullpath = os.path.join(target_folder, img_name)
            cv2.imwrite(img_fullpath, image[:, :, i] * 255)

    ref1_result_distributor: Distributor = Distributor(
        resize_to=input_sizes[2],
        image_transform_function=input_ref1_label_preprocessing_function,
        # each_transformed_image_save_function_optional=toolz.curry(
        #     save_batch_transformed_img2
        # )(training_result_folder, "training_ref1_result_"),
    )
    ref2_result_distributor: Distributor = Distributor(
        resize_to=input_sizes[3],
        image_transform_function=input_ref2_label_preprocessing_function,
        # each_transformed_image_save_function_optional=toolz.curry(
        #     save_batch_transformed_img2
        # )(training_result_folder, "training_ref2_result_"),
    )
    ref3_result_distributor: Distributor = Distributor(
        resize_to=input_sizes[4],
        image_transform_function=input_ref3_label_preprocessing_function,
        # each_transformed_image_save_function_optional=toolz.curry(
        #     save_batch_transformed_img2
        # )(training_result_folder, "training_ref3_result_"),
    )

    predict_ref_result_flow_manager = __input_result_label_flow(
        dataset_directory=predict_ref_result_label_folder,
        batch_size=predict_batch_size,
        shuffle=False,
        distributors=[
            ref1_result_distributor,
            ref2_result_distributor,
            ref3_result_distributor,
        ],
    )

    # 2.3 Inout ---------
    def __inout_dist_generator_infos(
        input_flows: List[DistFlowManager],
        output_flows: List[DistFlowManager],
        batch_size: int,
    ) -> Tuple[Generator, int, int, List[str]]:
        _in_out_generator = DistInOutGenerator(
            input_flows=input_flows, output_flows=output_flows,
        )

        _generator = _in_out_generator.get_generator()
        _samples = _in_out_generator.get_samples()
        _filenames = _in_out_generator.get_filenames()
        _nb_samples = math.ceil(_samples / batch_size)

        return _generator, _samples, _filenames, _nb_samples

    (
        predict_generator,
        predict_samples,
        predict_filenames,
        predict_nb_samples,
    ) = __inout_dist_generator_infos(
        input_flows=[
            predict_main_image_flow_manager,
            predict_ref_image_flow_manager,
            predict_ref_result_flow_manager,
        ],
        output_flows=[],
        batch_size=predict_batch_size,
    )

    # 3. Predict
    # -----------
    # 3.1 Parameters ---------
    predict_steps = predict_nb_samples // predict_batch_size
    results = model.predict_generator(predict_generator, steps=predict_steps, verbose=1)

    # 4. Post Processing
    # ------------------
    # for index, result in enumerate(results):
    #     name: str = os.path.basename(predict_filenames[index])
    #     print("Post Processing for {}".format(name))
    #     full_path: str = os.path.join(predict_result_folder, name)
    #     result = ratio_to_img(result)
    #     cv2.imwrite(full_path, result)
