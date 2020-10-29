import os
import sys

sys.path.append(os.getcwd())

import time
from typing import List, Optional

import common_py
from common_py.dl.report import acc_loss_plot
from common_py.folder import files_in_folder
from image_keras.custom.callbacks_after_epoch import (
    EarlyStoppingAfter,
    ModelCheckpointAfter,
)
from tensorflow.keras.callbacks import Callback, History, TensorBoard

if __name__ == "__main__":
    # Variables
    from models.semantic_segmentation.unet_l4.config_004 import (
        UNetL4ConfigSequence,
        UnetL4ModelHelper,
    )

    variable_training_dataset_folder = "training_original_20_edge10"
    variable_validation_dataset_folder = "validation_original_20_edge10"
    variable_model_name = "unet_l4"
    variable_config_id = "004"
    variable_model_helper = UnetL4ModelHelper()

    # Continue training
    continue_tf_log_folder: Optional[str] = None
    continue_from_model: Optional[str] = None
    continue_initial_epoch: Optional[int] = None
    # continue_tf_log_folder: Optional[
    #     str
    # ] = "_training__model_ref_local_tracking_model_003__config_001__run_20201028-131347"
    # continue_from_model: Optional[
    #     str
    # ] = "training__model_ref_local_tracking_model_003__config_001__run_20201028-131347.epoch_08-val_loss_0.096-val_acc_0.976.hdf5"
    # continue_initial_epoch: Optional[int] = 8

    # 0. Prepare
    # ----------
    # training_id: 사용한 모델, Training 날짜
    # 0.1 ID ---------
    model_name: str = variable_model_name
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    config_id: str = variable_config_id
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

    # continue setting (tf log)
    if continue_tf_log_folder is not None:
        run_log_dir = os.path.join(tf_log_folder, continue_tf_log_folder)

    # b) dataset folders
    training_dataset_folder: str = os.path.join(
        base_dataset_folder, variable_training_dataset_folder
    )
    val_dataset_folder: str = os.path.join(
        base_dataset_folder, variable_validation_dataset_folder
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
    model_helper = variable_model_helper

    # a) model (from python code)
    model = model_helper.get_model()

    # b) compile
    model = model_helper.compile_model(model)

    # continue setting (weights)
    if continue_from_model is not None:
        model.load_weights(os.path.join(save_weights_folder, continue_from_model))

    # 2. Dataset
    # ----------
    training_batch_size: int = 8
    val_batch_size: int = 8
    test_batch_size: int = 8

    # 2.1 Training, Validation Generator ---------
    training_images_file_names = files_in_folder(training_image_folder)
    training_samples = len(training_images_file_names)
    training_sequence = UNetL4ConfigSequence(
        image_file_names=training_images_file_names,
        main_image_folder_name=training_image_folder,
        output_label_folder_name=training_label_folder,
        batch_size=training_batch_size,
        shuffle=True,
    )

    val_images_file_names = files_in_folder(val_image_folder)
    val_samples = len(val_images_file_names)
    val_sequence = UNetL4ConfigSequence(
        image_file_names=val_images_file_names,
        main_image_folder_name=val_image_folder,
        output_label_folder_name=val_label_folder,
        batch_size=val_batch_size,
        shuffle=False,
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

    val_metric = model.compiled_metrics._metrics[1].name
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
    tensorboard_cb: Callback = TensorBoard(log_dir=run_log_dir, write_images=True)
    callback_list: List[Callback] = [tensorboard_cb, model_checkpoint, early_stopping]

    # 3.3 Training ---------
    # continue setting (initial epoch)
    initial_epoch = 0
    if continue_initial_epoch is not None:
        initial_epoch = continue_initial_epoch

    history: History = model.fit(
        training_sequence,
        epochs=training_num_of_epochs,
        verbose=1,
        callbacks=callback_list,
        validation_data=val_sequence,
        shuffle=True,
        initial_epoch=initial_epoch,
        steps_per_epoch=training_steps_per_epoch,
        validation_steps=val_steps,
        validation_freq=val_freq,
        max_queue_size=10,
        workers=8,
        use_multiprocessing=True,
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
