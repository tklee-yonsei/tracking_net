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
    output_label_preprocessing_function,
)

if __name__ == "__main__":
    # 0. Prepare
    # ----------

    # test_id: 사용한 모델, Test 날짜
    # 0.1 ID ---------
    model_name: str = "model_006"
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
    test_dataset_folder: str = os.path.join(base_dataset_folder, "ivan_filtered_test")
    # input - main image
    test_main_image_folder: str = os.path.join(test_dataset_folder, "image", "current")
    # input - ref image
    test_ref_image_folder: str = os.path.join(test_dataset_folder, "image", "prev")
    # input - ref result label
    test_ref_result_label_folder: str = os.path.join(test_dataset_folder, "prev_result")
    # output - main label
    test_output_main_label_folder: str = os.path.join(test_dataset_folder, "label")

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
    test_batch_size: int = 4

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

    test_main_image_flow_manager = __input_main_image_flow(
        dataset_directory=test_main_image_folder,
        batch_size=test_batch_size,
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

    test_ref_image_flow_manager = __input_ref_image_flow(
        dataset_directory=test_ref_image_folder,
        batch_size=test_batch_size,
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
    output_helper_ref_result_distributor: Distributor = Distributor(
        resize_to=input_sizes[4]
    )

    test_ref_result_flow_manager = __input_result_label_flow(
        dataset_directory=test_ref_result_label_folder,
        batch_size=test_batch_size,
        shuffle=False,
        distributors=[
            ref1_result_distributor,
            ref2_result_distributor,
            ref3_result_distributor,
            output_helper_ref_result_distributor,
        ],
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
    ) -> DistFlowManager:
        _each_transformed_label_save_function_optional = None
        if save_folder_and_prefix is not None:
            _each_transformed_label_save_function_optional = toolz.curry(
                save_batch_transformed_img
            )(save_folder_and_prefix[0], save_folder_and_prefix[1])
        _label_flow: FlowFromDirectory = ImagesFromDirectory(
            dataset_directory=dataset_directory,
            batch_size=batch_size,
            color_mode="rgb",
            shuffle=shuffle,
            seed=seed,
        )
        _distributor: Distributor = Distributor(
            resize_to=output_sizes[0],
            image_transform_function=preprocessing_function,
            each_transformed_image_save_function_optional=_each_transformed_label_save_function_optional,
            # transform_function_for_all=img_to_ratio,
        )
        _image_flow_manager: DistFlowManager = DistFlowManager(
            flow_from_directory=_label_flow, distributors=[_distributor],
        )
        return _image_flow_manager

    test_output_main_label_flow_manager = __output_label_flow(
        dataset_directory=test_output_main_label_folder,
        batch_size=test_batch_size,
        shuffle=False,
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
        _nb_samples = math.ceil(_samples / batch_size)

        return _generator, _samples, _nb_samples

    (test_generator, test_samples, test_nb_samples,) = __inout_dist_generator_infos(
        input_flows=[
            test_main_image_flow_manager,
            test_ref_image_flow_manager,
            test_ref_result_flow_manager,
        ],
        output_flows=[test_output_main_label_flow_manager],
        batch_size=test_batch_size,
    )

    # 2.4 Custom Generator Transform ---------
    def _zipped_transform(zipped_inout):
        for idx, zipped_inout_element in enumerate(zipped_inout):
            zipped_in_element = zipped_inout_element[0]
            batch_size = zipped_in_element[0].shape[0]
            zipped_out_element = zipped_inout_element[1]

            def _modify_output(_in, _out, _batch_size):
                batch_out_list = []
                for i in range(_batch_size):
                    current_batch_in_list = _in[i]
                    current_batch_out_list = _out[i]
                    modified_out_list = output_label_preprocessing_function(
                        current_batch_out_list, current_batch_in_list
                    )
                    batch_out_list.append(modified_out_list)
                return np.array(batch_out_list)

            modified_output = _modify_output(
                zipped_in_element[5], zipped_out_element[0], batch_size
            )

            # for i in range(batch_size):
            #     save_batch_transformed_img2(
            #         training_result_folder,
            #         "training_output_main_label_",
            #         idx,
            #         i,
            #         modified_output[i],
            #     )

            yield (
                [
                    zipped_in_element[0],
                    zipped_in_element[1],
                    zipped_in_element[2],
                    zipped_in_element[3],
                    zipped_in_element[4],
                ],
                [modified_output],
            )

    test_generator2 = _zipped_transform(test_generator)

    # 3. Test
    # -----------
    # 3.1 Parameters ---------
    test_steps = test_samples // test_batch_size

    test_loss, test_acc = model.evaluate_generator(
        test_generator2, steps=test_steps, verbose=1, max_queue_size=1
    )

    metric_names = list(map(lambda el: el.name, model.metrics))

    print(
        "loss : {}".format(
            dict(map(lambda kv: (kv[0], kv[1].name), model.loss.items()))
        )
    )
    print("loss weights : {}".format(model.loss_weights))
    print("metrics : {}".format(list(map(lambda el: el.name, model.metrics))))

    print("test_loss: {}".format(test_loss))
    print("test_acc: {}".format(test_acc))
