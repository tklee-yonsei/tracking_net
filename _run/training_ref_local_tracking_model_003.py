import math
import os
import sys

from common_py.folder import files_in_folder

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
    import cv2
    import tensorflow as tf

    class RefTrackingSequence(tf.keras.utils.Sequence):
        def __init__(
            self,
            image_file_names: List[str],
            main_image_folder_name: str,
            ref_image_folder_name: str,
            ref_result_label_folder_name: str,
            output_label_folder_name: str,
            ref_result1_preprocessing_function: Callable[[np.ndarray], np.ndarray],
            ref_result2_preprocessing_function: Callable[[np.ndarray], np.ndarray],
            ref_result3_preprocessing_function: Callable[[np.ndarray], np.ndarray],
            ref_result4_preprocessing_function: Callable[[np.ndarray], np.ndarray],
            output_label_preprocessing_function: Callable[
                [np.ndarray, np.ndarray], np.ndarray
            ],
            main_image_preprocessing_function: Optional[
                Callable[[np.ndarray], np.ndarray]
            ] = None,
            ref_image_preprocessing_function: Optional[
                Callable[[np.ndarray], np.ndarray]
            ] = None,
            batch_size: int = 1,
            shuffle: bool = False,
            seed: int = 42,
        ):
            self.image_file_names = image_file_names
            self.main_image_folder_name = main_image_folder_name
            self.ref_image_folder_name = ref_image_folder_name
            self.ref_result_label_folder_name = ref_result_label_folder_name
            self.output_label_folder_name = output_label_folder_name
            self.main_image_preprocessing_function = main_image_preprocessing_function
            self.ref_image_preprocessing_function = ref_image_preprocessing_function
            self.ref_result1_preprocessing_function = ref_result1_preprocessing_function
            self.ref_result2_preprocessing_function = ref_result2_preprocessing_function
            self.ref_result3_preprocessing_function = ref_result3_preprocessing_function
            self.ref_result4_preprocessing_function = ref_result4_preprocessing_function
            self.output_label_preprocessing_function = (
                output_label_preprocessing_function
            )
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.seed = seed

        def __len__(self):
            return math.ceil(len(self.image_file_names) / self.batch_size)

        def __getitem__(self, idx):
            batch_image_file_names = self.image_file_names[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]

            batch_main_images = []
            batch_ref_images = []
            batch_ref1_results = []
            batch_ref2_results = []
            batch_ref3_results = []
            batch_ref4_results = []
            batch_output_labels = []
            for image_file_name in batch_image_file_names:
                main_img = cv2.imread(
                    os.path.join(self.main_image_folder_name, image_file_name),
                    cv2.IMREAD_GRAYSCALE,
                )
                if self.main_image_preprocessing_function is not None:
                    main_img = self.main_image_preprocessing_function(main_img)
                batch_main_images.append(main_img)

                ref_img = cv2.imread(
                    os.path.join(self.ref_image_folder_name, image_file_name),
                    cv2.IMREAD_GRAYSCALE,
                )
                if self.ref_image_preprocessing_function is not None:
                    ref_img = self.ref_image_preprocessing_function(ref_img)
                batch_ref_images.append(ref_img)

                ref_result_label = cv2.imread(
                    os.path.join(self.ref_result_label_folder_name, image_file_name)
                )

                ref_result1 = self.ref_result1_preprocessing_function(ref_result_label)
                batch_ref1_results.append(ref_result1)

                ref_result2 = self.ref_result2_preprocessing_function(ref_result_label)
                batch_ref2_results.append(ref_result2)

                ref_result3 = self.ref_result3_preprocessing_function(ref_result_label)
                batch_ref3_results.append(ref_result3)

                ref_result4 = self.ref_result4_preprocessing_function(ref_result_label)
                batch_ref4_results.append(ref_result4)

                output_label = cv2.imread(
                    os.path.join(self.output_label_folder_name, image_file_name)
                )
                output_label = self.output_label_preprocessing_function(
                    output_label, ref_result_label,
                )
                batch_output_labels.append(output_label)

            X = [
                np.array(batch_main_images),
                np.array(batch_ref_images),
                np.array(batch_ref1_results),
                np.array(batch_ref2_results),
                np.array(batch_ref3_results),
                np.array(batch_ref4_results),
            ]
            Y = [np.array(batch_output_labels)]

            return (X, Y)

    training_images_file_names = files_in_folder(training_main_image_folder)
    training_samples = len(training_images_file_names)
    training_sequence = RefTrackingSequence(
        image_file_names=training_images_file_names,
        main_image_folder_name=training_main_image_folder,
        ref_image_folder_name=training_ref_image_folder,
        ref_result_label_folder_name=training_ref_result_label_folder,
        output_label_folder_name=training_output_main_label_folder,
        main_image_preprocessing_function=toolz.compose_left(
            input_main_image_preprocessing_function, img_to_ratio
        ),
        ref_image_preprocessing_function=toolz.compose_left(
            input_ref_image_preprocessing_function, img_to_ratio
        ),
        ref_result1_preprocessing_function=input_ref_label_1_preprocessing_function,
        ref_result2_preprocessing_function=input_ref_label_2_preprocessing_function,
        ref_result3_preprocessing_function=input_ref_label_3_preprocessing_function,
        ref_result4_preprocessing_function=input_ref_label_4_preprocessing_function,
        output_label_preprocessing_function=output_label_preprocessing_function,
        batch_size=training_batch_size,
        shuffle=True,
    )

    val_images_file_names = files_in_folder(val_main_image_folder)
    val_samples = len(val_images_file_names)
    val_sequence = RefTrackingSequence(
        image_file_names=val_images_file_names,
        main_image_folder_name=val_main_image_folder,
        ref_image_folder_name=val_ref_image_folder,
        ref_result_label_folder_name=val_ref_result_label_folder,
        output_label_folder_name=val_output_main_label_folder,
        main_image_preprocessing_function=toolz.compose_left(
            input_main_image_preprocessing_function, img_to_ratio
        ),
        ref_image_preprocessing_function=toolz.compose_left(
            input_ref_image_preprocessing_function, img_to_ratio
        ),
        ref_result1_preprocessing_function=input_ref_label_1_preprocessing_function,
        ref_result2_preprocessing_function=input_ref_label_2_preprocessing_function,
        ref_result3_preprocessing_function=input_ref_label_3_preprocessing_function,
        ref_result4_preprocessing_function=input_ref_label_4_preprocessing_function,
        output_label_preprocessing_function=output_label_preprocessing_function,
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
    history: History = model.fit(
        training_sequence,
        epochs=training_num_of_epochs,
        verbose=1,
        callbacks=callback_list,
        validation_data=val_sequence,
        shuffle=True,
        initial_epoch=0,
        steps_per_epoch=training_steps_per_epoch,
        validation_steps=val_steps,
        validation_freq=val_freq,
        max_queue_size=4,
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
