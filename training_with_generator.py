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
from keras import losses, optimizers
from keras.callbacks import Callback, History

from idl.batch_transform import generate_iterator_and_transform
from idl.callbacks_after_epoch import EarlyStoppingAfter, ModelCheckpointAfter
from idl.flow_directory import FlowFromDirectory, ImagesFromDirectory
from idl.metrics import binary_class_mean_iou
from idl.model_io import load_model
from utils.image_transform import (
    gray_image_apply_clahe,
    img_resize,
    img_to_minmax,
    img_to_ratio,
)

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
    # training_id: 사용한 모델, Training 날짜
    model_name: str = "unet_l4"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    training_id: str = "_training__model_{}__run_{}".format(model_name, run_id)

    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")

    common_py.create_folder(save_models_folder)
    common_py.create_folder(save_weights_folder)

    training_dataset_folder: str = os.path.join(
        base_data_folder, "training_original_20_edge10"
    )
    validation_dataset_folder: str = os.path.join(
        base_data_folder, "validation_original_20_edge10"
    )

    # Generators
    # ----------
    batch_size: int = 8
    val_batch_size: int = 8
    test_batch_size: int = 8

    # training generator
    training_image_folder: str = os.path.join(training_dataset_folder, "image")
    training_label_folder: str = os.path.join(training_dataset_folder, "label")

    training_img_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=training_image_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        seed=42,
    )
    training_image_generator = training_img_flow.get_iterator()
    training_image_transformed_generator = generate_iterator_and_transform(
        image_generator=training_image_generator,
        each_image_transform_function=toolz.compose_left(
            lambda _img: np.array(_img, dtype=np.uint8),
            lambda _img: img_resize(_img, (256, 256)),
            gray_image_apply_clahe,
            lambda _img: np.reshape(_img, (_img.shape[0], _img.shape[1], 1)),
        ),
        transform_function_for_all=img_to_ratio,
    )

    training_label_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=training_label_folder,
        batch_size=batch_size,
        color_mode="grayscale",
        shuffle=True,
        seed=42,
    )
    training_label_generator = training_label_flow.get_iterator()
    training_label_transformed_generator = generate_iterator_and_transform(
        image_generator=training_label_generator,
        each_image_transform_function=(
            toolz.compose_left(
                lambda _img: np.array(_img, dtype=np.uint8),
                lambda _img: img_resize(_img, (256, 256)),
                lambda _img: img_to_minmax(_img, 127, (0, 255)),
                lambda _img: np.reshape(_img, (_img.shape[0], _img.shape[1], 1)),
            ),
            None,
        ),
        transform_function_for_all=img_to_ratio,
    )

    training_input_generator = map(list, zip(training_image_transformed_generator))
    training_output_generator = map(list, zip(training_label_transformed_generator))
    training_generator = zip(training_input_generator, training_output_generator)

    # validation generator
    validation_image_folder: str = os.path.join(validation_dataset_folder, "image")
    validation_label_folder: str = os.path.join(validation_dataset_folder, "label")

    validation_img_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=validation_image_folder,
        batch_size=val_batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    validation_image_generator = validation_img_flow.get_iterator()
    validation_image_transformed_generator = generate_iterator_and_transform(
        image_generator=validation_image_generator,
        each_image_transform_function=(
            toolz.compose_left(
                lambda _img: np.array(_img, dtype=np.uint8),
                lambda _img: img_resize(_img, (256, 256)),
                gray_image_apply_clahe,
                lambda _img: np.reshape(_img, (_img.shape[0], _img.shape[1], 1)),
            ),
            None,
        ),
        transform_function_for_all=img_to_ratio,
    )

    validation_label_flow: FlowFromDirectory = ImagesFromDirectory(
        dataset_directory=validation_label_folder,
        batch_size=val_batch_size,
        color_mode="grayscale",
        shuffle=False,
    )
    validation_label_generator = validation_label_flow.get_iterator()
    validation_label_transformed_generator = generate_iterator_and_transform(
        image_generator=validation_label_generator,
        each_image_transform_function=(
            toolz.compose_left(
                lambda _img: np.array(_img, dtype=np.uint8),
                lambda _img: img_resize(_img, (256, 256)),
                lambda _img: img_to_minmax(_img, 127, (0, 255)),
                lambda _img: np.reshape(_img, (_img.shape[0], _img.shape[1], 1)),
            ),
            None,
        ),
        transform_function_for_all=img_to_ratio,
    )

    validation_input_generator = map(list, zip(validation_image_transformed_generator))
    validation_output_generator = map(list, zip(validation_label_transformed_generator))
    validation_generator = zip(validation_input_generator, validation_output_generator)

    # Training
    # --------
    training_num_of_epochs: int = 200
    training_samples: int = training_image_generator.samples
    training_steps_per_epoch: int = training_samples // batch_size

    val_freq: int = 1
    val_samples: int = validation_image_generator.samples
    val_steps: int = val_samples // val_batch_size

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

    # model
    model_path: str = os.path.join(save_models_folder, "unet_l4_000.json")
    model: keras.models.Model = load_model(model_path)

    model.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss=losses.binary_crossentropy,
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), binary_class_mean_iou],
    )

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

    history_target_folder, acc_plot_image_name, loss_plot_image_name = acc_loss_plot(
        history.history["accuracy"],
        history.history["loss"],
        history.history["val_accuracy"],
        history.history["val_loss"],
        training_id[1:],
        save_weights_folder,
    )
