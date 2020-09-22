import math
import os
import time
from typing import Generator, List, Tuple

import common_py
import toolz
from common_py.dl.report import acc_loss_plot
from image_keras.custom.callbacks_after_epoch import (
    EarlyStoppingAfter,
    ModelCheckpointAfter,
)
from image_keras.flow_directory import FlowFromDirectory, ImagesFromDirectory
from image_keras.inout_generator import BaseInOutGenerator, FlowManager
from image_keras.model_io import load_model
from image_keras.utils.image_transform import (
    gray_image_apply_clahe,
    img_to_minmax,
    img_to_ratio,
)
from keras.callbacks import Callback, History

from models.semantic_segmentation.unet_l4.unet_l4 import UnetL4ModelHelper

if __name__ == "__main__":
    # 0. Prepare
    # ----------

    # training_id: 사용한 모델, Training 날짜
    # 0.1 ID ---------
    model_name: str = "unet_l4"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    training_id: str = "_training__model_{}__run_{}".format(model_name, run_id)

    # 0.2 Folder ---------

    # a) model, weights, result
    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    common_py.create_folder(save_models_folder)
    common_py.create_folder(save_weights_folder)

    # b) dataset folders
    training_dataset_folder: str = os.path.join(
        base_data_folder, "training_original_20_edge10"
    )
    val_dataset_folder: str = os.path.join(
        base_data_folder, "validation_original_20_edge10"
    )
    # input - image
    training_image_folder: str = os.path.join(training_dataset_folder, "image")
    val_image_folder: str = os.path.join(val_dataset_folder, "image")
    # output - label
    training_label_folder: str = os.path.join(training_dataset_folder, "label")
    val_label_folder: str = os.path.join(val_dataset_folder, "label")

    # 1. Model
    # --------
    # model -> compile
    model_helper = UnetL4ModelHelper()

    # a) model (from python code)
    model = model_helper.get_model()

    # a) model (from json)
    # model_path: str = os.path.join(save_models_folder, "unet_l4_001.json")
    # model = load_model(model_path)

    # b) compile
    model = model_helper.compile_model(model)

    # 2. Dataset
    # ----------
    training_batch_size: int = 8
    val_batch_size: int = 8
    test_batch_size: int = 8

    # 2.1 Input ---------
    input_sizes = model_helper.model_descriptor.get_input_sizes()

    # a) image
    def __input_image_flow(
        dataset_directory: str, batch_size: int, shuffle: bool, seed: int = 42
    ) -> FlowManager:
        _img_flow: FlowFromDirectory = ImagesFromDirectory(
            dataset_directory=dataset_directory,
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle=shuffle,
            seed=seed,
        )
        _each_image_transform_function = gray_image_apply_clahe
        _image_flow_manager: FlowManager = FlowManager(
            flow_from_directory=_img_flow,
            resize_to=input_sizes[0],
            image_transform_function=_each_image_transform_function,
            each_transformed_image_save_function_optional=None,
            transform_function_for_all=img_to_ratio,
        )
        return _image_flow_manager

    training_image_flow_manager = __input_image_flow(
        dataset_directory=training_image_folder,
        batch_size=training_batch_size,
        shuffle=True,
    )
    val_image_flow_manager = __input_image_flow(
        dataset_directory=val_image_folder, batch_size=val_batch_size, shuffle=False
    )

    # 2.2 Output ---------
    output_sizes = model_helper.model_descriptor.get_output_sizes()

    # a) label
    def __output_label_flow(
        dataset_directory: str, batch_size: int, shuffle: bool, seed: int = 42
    ) -> FlowManager:
        _label_flow: FlowFromDirectory = ImagesFromDirectory(
            dataset_directory=dataset_directory,
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle=shuffle,
            seed=seed,
        )
        _each_label_transform_function = toolz.compose_left(
            lambda _img: img_to_minmax(_img, 127, (0, 255)),
        )
        _label_flow_manager: FlowManager = FlowManager(
            flow_from_directory=_label_flow,
            resize_to=output_sizes[0],
            image_transform_function=_each_label_transform_function,
            each_transformed_image_save_function_optional=None,
            transform_function_for_all=img_to_ratio,
        )
        return _label_flow_manager

    training_label_flow_manager = __output_label_flow(
        dataset_directory=training_label_folder,
        batch_size=training_batch_size,
        shuffle=True,
    )
    val_label_flow_manager = __output_label_flow(
        dataset_directory=val_label_folder, batch_size=val_batch_size, shuffle=False
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
        _nb_samples = math.ceil(_samples / training_batch_size)

        return _generator, _samples, _nb_samples

    (
        training_generator,
        training_samples,
        training_nb_samples,
    ) = __inout_generator_infos(
        input_flows=[training_image_flow_manager],
        output_flows=[training_label_flow_manager],
    )
    (val_generator, val_samples, val_nb_samples) = __inout_generator_infos(
        input_flows=[val_image_flow_manager], output_flows=[val_label_flow_manager],
    )

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

    model_checkpoint: Callback = ModelCheckpointAfter(
        os.path.join(
            save_weights_folder,
            training_id[1:]
            + ".epoch_{epoch:02d}-val_loss_{val_loss:.3f}-val_mean_iou_{val_mean_iou:.3f}.hdf5",
        ),
        verbose=1,
        after_epoch=apply_callbacks_after,
    )
    early_stopping: Callback = EarlyStoppingAfter(
        patience=early_stopping_patience, verbose=1, after_epoch=apply_callbacks_after,
    )
    callback_list: List[Callback] = [model_checkpoint, early_stopping]

    # 3.3 Training ---------
    history: History = model.fit_generator(
        training_generator,
        callbacks=callback_list,
        steps_per_epoch=training_steps_per_epoch,
        epochs=training_num_of_epochs,
        verbose=1,
        validation_data=val_generator,
        validation_steps=val_steps,
        validation_freq=val_freq,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0,
    )

    # History display
    history_target_folder, acc_plot_image_name, loss_plot_image_name = acc_loss_plot(
        history.history["accuracy"],
        history.history["loss"],
        history.history["val_accuracy"],
        history.history["val_loss"],
        training_id[1:],
        save_weights_folder,
    )
