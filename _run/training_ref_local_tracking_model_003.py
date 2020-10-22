import math
import os
import sys

sys.path.append(os.getcwd())

import time
from typing import Callable, Generator, List, Optional, Tuple

import common_py
import numpy as np
import toolz
from common_py.dl.report import acc_loss_plot
from image_keras.custom.callbacks_after_epoch import (
    EarlyStoppingAfter,
    ModelCheckpointAfter,
)
from image_keras.flow_directory import FlowFromDirectory, ImagesFromDirectory
from image_keras.inout_generator import (
    DistFlowManager,
    DistInOutGenerator,
    Distributor,
    save_batch_transformed_img,
)
from image_keras.utils.image_transform import img_to_ratio
from tensorflow.keras.callbacks import Callback, History, TensorBoard

if __name__ == "__main__":
    # Variables
    from models.ref_local_tracking.ref_local_tracking_model_003 import (
        RefModel003ModelHelper,
        input_main_image_preprocessing_function,
        input_ref_image_preprocessing_function,
        input_ref_label_1_preprocessing_function,
        input_ref_label_2_preprocessing_function,
        input_ref_label_3_preprocessing_function,
        input_ref_label_4_preprocessing_function,
        output_label_preprocessing_function,
    )

    variable_training_dataset_folder = "ivan_filtered_training"
    variable_validation_dataset_folder = "ivan_filtered_validation"
    variable_model_name = "ref_local_tracking_model_003"
    variable_config_id = "001"

    from models.semantic_segmentation.unet_l4.config_005 import UnetL4ModelHelper

    variable_unet_weights_file_name = "training__model_unet_l4__config_005__run_20201021-141844.epoch_25-val_loss_0.150-val_mean_iou_0.931.hdf5"
    # 0. Prepare
    # ----------

    # training_id: 사용한 모델, Training 날짜
    # 0.1 ID ---------
    model_name: str = variable_model_name
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    config_id = variable_config_id
    training_id: str = "_training__model_{}__config_{}__run_{}".format(
        model_name, config_id, run_id
    )
    print("# Information ---------------------------")
    print("Training ID: {}".format(training_id))
    print("Training Dataset: {}".format(variable_training_dataset_folder))
    print("Validation Dataset: {}".format(variable_validation_dataset_folder))
    print("Config ID: {}".format(variable_config_id))
    print("-----------------------------------------")

    # 0.2 Folder ---------
    # a) model, weights, result
    base_dataset_folder: str = os.path.join("dataset")
    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    tf_log_folder: str = os.path.join(base_save_folder, "tf_logs")
    common_py.create_folder(save_models_folder)
    common_py.create_folder(save_weights_folder)
    common_py.create_folder(tf_log_folder)
    run_log_dir: str = os.path.join(tf_log_folder, training_id)
    training_result_folder: str = os.path.join(base_data_folder, training_id)
    common_py.create_folder(training_result_folder)

    # b) dataset folders
    training_dataset_folder: str = os.path.join(
        base_dataset_folder, variable_training_dataset_folder
    )
    val_dataset_folder: str = os.path.join(
        base_dataset_folder, variable_validation_dataset_folder
    )
    # input - main image
    training_main_image_folder: str = os.path.join(
        training_dataset_folder, "image", "current"
    )
    val_main_image_folder: str = os.path.join(val_dataset_folder, "image", "current")
    # input - ref image
    training_ref_image_folder: str = os.path.join(
        training_dataset_folder, "image", "prev"
    )
    val_ref_image_folder: str = os.path.join(val_dataset_folder, "image", "prev")
    # input - ref result label
    training_ref_result_label_folder: str = os.path.join(
        training_dataset_folder, "prev_result"
    )
    val_ref_result_label_folder: str = os.path.join(val_dataset_folder, "prev_result")
    # output - main label
    training_output_main_label_folder: str = os.path.join(
        training_dataset_folder, "label"
    )
    val_output_main_label_folder: str = os.path.join(val_dataset_folder, "label")

    # 1. Model
    # --------
    # model -> compile
    unet_model_helper = UnetL4ModelHelper()
    unet_model = unet_model_helper.get_model()
    unet_model_weights_path: str = os.path.join(
        save_weights_folder, variable_unet_weights_file_name
    )
    unet_model.load_weights(unet_model_weights_path)
    model_helper = RefModel003ModelHelper(pre_trained_unet_l4_model=unet_model)

    # a) model (from python code)
    model = model_helper.get_model()

    # b) compile
    model = model_helper.compile_model(model)

    # 2. Dataset
    # ----------
    training_batch_size: int = 4
    val_batch_size: int = 4
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

    training_main_image_flow_manager = __input_main_image_flow(
        dataset_directory=training_main_image_folder,
        batch_size=training_batch_size,
        preprocessing_function=input_main_image_preprocessing_function,
        # save_folder_and_prefix=(training_result_folder, "training_main_image_"),
        shuffle=True,
    )
    val_main_image_flow_manager = __input_main_image_flow(
        dataset_directory=val_main_image_folder,
        batch_size=val_batch_size,
        preprocessing_function=input_main_image_preprocessing_function,
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

    training_ref_image_flow_manager = __input_ref_image_flow(
        dataset_directory=training_ref_image_folder,
        batch_size=training_batch_size,
        preprocessing_function=input_ref_image_preprocessing_function,
        # save_folder_and_prefix=(training_result_folder, "training_ref_image_"),
        shuffle=True,
    )
    val_ref_image_flow_manager = __input_ref_image_flow(
        dataset_directory=val_ref_image_folder,
        batch_size=val_batch_size,
        preprocessing_function=input_ref_image_preprocessing_function,
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
        resize_to=input_sizes[0],
        image_transform_function=input_ref_label_1_preprocessing_function,
        # each_transformed_image_save_function_optional=toolz.curry(
        #     save_batch_transformed_img2
        # )(training_result_folder, "training_ref1_result_"),
    )
    ref2_result_distributor: Distributor = Distributor(
        resize_to=input_sizes[0],
        image_transform_function=input_ref_label_2_preprocessing_function,
        # each_transformed_image_save_function_optional=toolz.curry(
        #     save_batch_transformed_img2
        # )(training_result_folder, "training_ref2_result_"),
    )
    ref3_result_distributor: Distributor = Distributor(
        resize_to=input_sizes[0],
        image_transform_function=input_ref_label_3_preprocessing_function,
        # each_transformed_image_save_function_optional=toolz.curry(
        #     save_batch_transformed_img2
        # )(training_result_folder, "training_ref3_result_"),
    )
    ref4_result_distributor: Distributor = Distributor(
        resize_to=input_sizes[0],
        image_transform_function=input_ref_label_4_preprocessing_function,
        # each_transformed_image_save_function_optional=toolz.curry(
        #     save_batch_transformed_img2
        # )(training_result_folder, "training_ref3_result_"),
    )
    output_helper_ref_result_distributor: Distributor = Distributor(
        resize_to=input_sizes[0],
        # image_transform_function=lambda im: im,
        # each_transformed_image_save_function_optional=toolz.curry(
        #     save_batch_transformed_img
        # )(training_result_folder, "training_output_helper_ref_"),
    )

    training_ref_result_flow_manager = __input_result_label_flow(
        dataset_directory=training_ref_result_label_folder,
        batch_size=training_batch_size,
        shuffle=True,
        distributors=[
            ref1_result_distributor,
            ref2_result_distributor,
            ref3_result_distributor,
            ref4_result_distributor,
            output_helper_ref_result_distributor,
        ],
    )
    val_ref_result_flow_manager = __input_result_label_flow(
        dataset_directory=val_ref_result_label_folder,
        batch_size=val_batch_size,
        shuffle=False,
        distributors=[
            ref1_result_distributor,
            ref2_result_distributor,
            ref3_result_distributor,
            ref4_result_distributor,
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

    training_output_main_label_flow_manager = __output_label_flow(
        dataset_directory=training_output_main_label_folder,
        batch_size=training_batch_size,
        shuffle=True,
    )
    val_output_main_label_flow_manager = __output_label_flow(
        dataset_directory=val_output_main_label_folder,
        batch_size=val_batch_size,
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

    (
        training_generator,
        training_samples,
        training_nb_samples,
    ) = __inout_dist_generator_infos(
        input_flows=[
            training_main_image_flow_manager,
            training_ref_image_flow_manager,
            training_ref_result_flow_manager,
        ],
        output_flows=[training_output_main_label_flow_manager],
        batch_size=training_batch_size,
    )
    (val_generator, val_samples, val_nb_samples) = __inout_dist_generator_infos(
        input_flows=[
            val_main_image_flow_manager,
            val_ref_image_flow_manager,
            val_ref_result_flow_manager,
        ],
        output_flows=[val_output_main_label_flow_manager],
        batch_size=val_batch_size,
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
                zipped_in_element[6], zipped_out_element[0], batch_size
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
                    zipped_in_element[5],
                ],
                [modified_output],
            )

    training_generator2 = _zipped_transform(training_generator)
    val_generator2 = _zipped_transform(val_generator)

    # 3. Training
    # -----------
    # 3.1 Parameters ---------
    training_num_of_epochs: int = 200
    training_steps_per_epoch: int = training_samples // training_batch_size

    val_freq: int = 1
    val_steps: int = val_samples // val_batch_size

    # 3.2 Callbacks ---------
    apply_callbacks_after: int = 0
    early_stopping_patience: int = training_num_of_epochs // (10 * val_freq)
    val_metric = model.compiled_metrics._metrics[-1].name
    val_checkpoint_metric = "val_" + val_metric
    model_checkpoint: Callback = ModelCheckpointAfter(
        os.path.join(
            save_weights_folder,
            training_id[1:]
            + ".epoch_{epoch:02d}-val_loss_{val_loss:.3f}-"
            + val_checkpoint_metric
            + "_{"
            + val_checkpoint_metric
            + ":.3f}.hdf5",
            # + ".epoch_{epoch:02d}-val_loss_{val_loss:.3f}-val_mean_iou_{val_mean_iou:.3f}.hdf5",
        ),
        verbose=1,
        after_epoch=apply_callbacks_after,
    )
    early_stopping: Callback = EarlyStoppingAfter(
        patience=early_stopping_patience, verbose=1, after_epoch=apply_callbacks_after,
    )
    tensorboard_cb: Callback = TensorBoard(log_dir=run_log_dir)
    callback_list: List[Callback] = [tensorboard_cb, model_checkpoint, early_stopping]

    # 3.3 Training ---------
    history: History = model.fit_generator(
        training_generator2,
        callbacks=callback_list,
        steps_per_epoch=training_steps_per_epoch,
        epochs=training_num_of_epochs,
        verbose=1,
        validation_data=val_generator2,
        validation_steps=val_steps,
        validation_freq=val_freq,
        max_queue_size=2,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0,
    )

    # History display
    history_target_folder, acc_plot_image_name, loss_plot_image_name = acc_loss_plot(
        history.history[val_metric],
        history.history["loss"],
        history.history["val_{}".format(val_metric)],
        history.history["val_loss"],
        training_id[1:],
        save_weights_folder,
    )
