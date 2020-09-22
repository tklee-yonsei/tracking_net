import os
import time
from typing import List

import common_py
import cv2
import keras
import numpy as np
import tensorflow as tf
import toolz
from common_py.dl.report import acc_loss_plot
from image_keras.batch_transform import generate_iterator_and_transform
from image_keras.custom.callbacks_after_epoch import (
    EarlyStoppingAfter,
    ModelCheckpointAfter,
)
from image_keras.custom.metrics import binary_class_mean_iou
from image_keras.flow_directory import FlowFromDirectory, ImagesFromDirectory
from image_keras.inout_generator import BaseInOutGenerator, FlowManager
from image_keras.model_io import load_model
from image_keras.utils.image_transform import (
    gray_image_apply_clahe,
    img_resize,
    img_to_minmax,
    img_to_ratio,
)
from keras import losses, optimizers
from keras.callbacks import Callback, History

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
    index_num: int, batch_num: int, transformed_batch_img: np.ndarray
) -> None:
    # 이름
    img_path = os.path.join(".")
    img_name = "img_transformed_{:04d}_{:02d}.png".format(index_num, batch_num)
    img_fullpath = os.path.join(img_path, img_name)

    # 저장
    cv2.imwrite(img_fullpath, transformed_batch_img)


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

    # Dataset Generator
    # -----------------
    training_batch_size: int = 8
    val_batch_size: int = 8
    test_batch_size: int = 8

    # 1) Training
    # image
    # training_image_folder: str = os.path.join(training_dataset_folder, "image")
    training_img_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=training_image_folder,
        batch_size=training_batch_size,
        color_mode="grayscale",
        shuffle=True,
        seed=42,
    )
    training_each_image_transform_function = toolz.compose_left(
        lambda _img: img_resize(_img, (256, 256)), gray_image_apply_clahe,
    )
    training_image_flow_manager: FlowManager = FlowManager(
        flow_from_directory=training_img_flow,
        resize_to=(256, 256),
        image_transform_function=training_each_image_transform_function,
        transform_function_for_all=img_to_ratio,
    )

    # label
    # training_label_folder: str = os.path.join(training_dataset_folder, "label")

    training_label_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=training_label_folder,
        batch_size=training_batch_size,
        color_mode="grayscale",
        shuffle=True,
        seed=42,
    )
    training_each_label_transform_function = toolz.compose_left(
        lambda _img: img_resize(_img, (256, 256)),
        lambda _img: img_to_minmax(_img, 127, (0, 255)),
    )
    training_label_flow_manager: FlowManager = FlowManager(
        flow_from_directory=training_label_flow,
        resize_to=(256, 256),
        image_transform_function=training_each_label_transform_function,
        transform_function_for_all=img_to_ratio,
    )

    # inout
    training_in_out_generator = BaseInOutGenerator(
        input_flows=[training_image_flow_manager],
        output_flows=[training_label_flow_manager],
    )
    training_generator = training_in_out_generator.get_generator()

    # 2) Validation
    # image
    # validation_image_folder: str = os.path.join(validation_dataset_folder, "image")
    validation_img_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=val_image_folder,
        batch_size=val_batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    validation_each_image_transform_function = toolz.compose_left(
        lambda _img: img_resize(_img, (256, 256)), gray_image_apply_clahe,
    )
    validation_image_flow_manager: FlowManager = FlowManager(
        flow_from_directory=validation_img_flow,
        resize_to=(256, 256),
        image_transform_function=validation_each_image_transform_function,
        transform_function_for_all=img_to_ratio,
    )

    # label
    # validation_label_folder: str = os.path.join(validation_dataset_folder, "label")

    validation_label_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=val_label_folder,
        batch_size=val_batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    validation_each_label_transform_function = toolz.compose_left(
        lambda _img: img_resize(_img, (256, 256)),
        lambda _img: img_to_minmax(_img, 127, (0, 255)),
    )
    validation_label_flow_manager: FlowManager = FlowManager(
        flow_from_directory=validation_label_flow,
        resize_to=(256, 256),
        image_transform_function=validation_each_label_transform_function,
        transform_function_for_all=img_to_ratio,
    )

    # inout
    validation_in_out_generator = BaseInOutGenerator(
        input_flows=[validation_image_flow_manager],
        output_flows=[validation_label_flow_manager],
    )
    validation_generator = validation_in_out_generator.get_generator()

    # Training
    # --------
    # 1) Sample, Step
    training_num_of_epochs: int = 200
    training_samples: int = training_in_out_generator.get_samples()
    training_steps_per_epoch: int = training_samples // training_batch_size

    val_freq: int = 1
    val_samples: int = validation_in_out_generator.get_samples()
    val_steps: int = val_samples // val_batch_size

    # 2) Callbacks
    apply_callbacks_after: int = 0
    early_stopping_patience: int = training_num_of_epochs // (10 * val_freq)

    model_checkpoint: Callback = ModelCheckpointAfter(
        os.path.join(
            save_weights_folder,
            training_id[1:]
            + ".epoch_{epoch:02d}-val_loss_{val_loss:.3f}-val_mean_iou_{val_mean_iou:.3f}.hdf5",
        ),
        verbose=1,
        # save_best_only=True,
        after_epoch=apply_callbacks_after,
    )
    early_stopping: Callback = EarlyStoppingAfter(
        patience=early_stopping_patience, verbose=1, after_epoch=apply_callbacks_after,
    )
    callback_list: List[Callback] = [model_checkpoint, early_stopping]

    # 3) Model
    from models.semantic_segmentation.unet_l4.unet_l4 import UnetL4ModelHelper

    model_helper = UnetL4ModelHelper()
    model_path: str = os.path.join(save_models_folder, "unet_l4_001.json")
    model: keras.models.Model = load_model(model_path)
    # model = model_helper.get_model()
    model = model_helper.compile_model(model)
    model.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss=losses.binary_crossentropy,
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), binary_class_mean_iou],
    )

    # 4) Training, Validation
    history: History = model.fit_generator(
        training_generator,
        callbacks=callback_list,
        steps_per_epoch=training_steps_per_epoch,
        epochs=training_num_of_epochs,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=val_steps,
        validation_freq=val_freq,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        shuffle=True,
        initial_epoch=0,
    )

    # 5) History
    history_target_folder, acc_plot_image_name, loss_plot_image_name = acc_loss_plot(
        history.history["accuracy"],
        history.history["loss"],
        history.history["val_accuracy"],
        history.history["val_loss"],
        training_id[1:],
        save_weights_folder,
    )
