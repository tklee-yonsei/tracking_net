import os
import sys

sys.path.append(os.getcwd())

import time
from typing import List, Optional

import common_py
import tensorflow as tf
from common_py.dl.report import acc_loss_plot
from common_py.folder import files_in_folder
from image_keras.custom.callbacks_after_epoch import (
    EarlyStoppingAfter,
    ModelCheckpointAfter,
)
from models.ref_local_tracking.ref_local_tracking_model_003.config import (
    RefTrackingSequence,
    tf_input_ref_label_1_preprocessing_function,
    tf_input_ref_label_2_preprocessing_function,
    tf_input_ref_label_3_preprocessing_function,
    tf_input_ref_label_4_preprocessing_function,
    tf_main_image_preprocessing_sequence,
    tf_output_label_processing,
    tf_ref_image_preprocessing_sequence,
)
from tensorflow.keras.callbacks import Callback, History, TensorBoard
from utils.gc_storage import get_file_from_google_bucket
from utils.gc_tpu import tpu_initialize
from utils.tf_images import decode_png

# tf.config.threading.set_inter_op_parallelism_threads(8)
# print(tf.config.threading.get_inter_op_parallelism_threads)

if __name__ == "__main__":
    # Variables
    from models.ref_local_tracking.ref_local_tracking_model_003 import (
        RefModel003ModelHelper,
    )

    variable_training_dataset_folder = "ivan_filtered_training"
    # variable_training_dataset_folder = "test_dataset"
    variable_validation_dataset_folder = "ivan_filtered_validation"
    variable_model_name = "ref_local_tracking_model_003"
    variable_config_id = "001"

    # from models.semantic_segmentation.unet_l4.config_005 import UnetL4ModelHelper

    # variable_unet_weights_file_name = "training__model_unet_l4__config_005__run_20201021-141844.epoch_25-val_loss_0.150-val_mean_iou_0.931.hdf5"

    from models.semantic_segmentation.unet_l4.config_001 import UnetL4ModelHelper

    variable_unet_weights_file_name = "unet010.hdf5"

    # gs_path = "gs://cell_tracking_dataset"
    gs_path = "gs://cell_tracking_dataset-1"
    tpu_name = "training-model-1"

    resolver = tpu_initialize(tpu_address=tpu_name)
    strategy = tf.distribute.TPUStrategy(resolver)

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
    os.environ["TZ"] = "Asia/Seoul"
    time.tzset()
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
    local_temp_folder: str = os.path.join("temp")
    common_py.create_folder(local_temp_folder)

    base_dataset_folder: str = os.path.join(gs_path, "dataset")
    base_data_folder: str = os.path.join(gs_path, "data")
    base_save_folder: str = os.path.join(gs_path, "save")
    # base_dataset_folder: str = os.path.join("dataset")
    # base_data_folder: str = os.path.join("data")
    # base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    tf_log_folder: str = os.path.join(base_save_folder, "tf_logs")
    common_py.create_folder(save_models_folder)
    common_py.create_folder(save_weights_folder)
    common_py.create_folder(tf_log_folder)
    run_log_dir: str = os.path.join(tf_log_folder, training_id)
    training_result_folder: str = os.path.join(base_data_folder, training_id)
    common_py.create_folder(training_result_folder)

    check_weight_name = training_id[1:]
    # continue setting (tf log)
    # if continue_tf_log_folder is not None:
    #     run_log_dir = os.path.join(tf_log_folder, continue_tf_log_folder)
    #     check_weight_name = continue_tf_log_folder[1:]

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
    with strategy.scope():
        # temp_unet_model_location = get_file_from_google_bucket(
        #     os.path.join(save_weights_folder, variable_unet_weights_file_name),
        #     os.path.join(local_temp_folder, variable_unet_weights_file_name),
        # )
        # unet_model_helper = UnetL4ModelHelper()
        # unet_model = unet_model_helper.get_model()
        # # unet_model.load_weights(temp_unet_model_location)
        # unet_model.load_weights("gs://cell_tracking_dataset-1/save/models/unet010")
        unet_model = tf.keras.models.load_model(
            "gs://cell_tracking_dataset-1/save/models/unet010"
        )

        # model_helper = RefModel003ModelHelper(pre_trained_unet_l4_model=unet_model)

        # # a) model (from python code)
        # model = model_helper.get_model()
        from models.ref_local_tracking.ref_local_tracking_model_003.model import (
            ref_local_tracking_model_003,
        )

        model = ref_local_tracking_model_003(
            pre_trained_unet_l4_model=unet_model,
            input_main_image_name="main_image",
            input_main_image_shape=(256, 256, 1),
            input_ref_image_name="ref_image",
            input_ref_image_shape=(256, 256, 1),
            input_ref_label_1_name="bin_label_1",
            input_ref_label_1_shape=(32, 32, 30),
            input_ref_label_2_name="bin_label_2",
            input_ref_label_2_shape=(64, 64, 30),
            input_ref_label_3_name="bin_label_3",
            input_ref_label_3_shape=(128, 128, 30),
            input_ref_label_4_name="bin_label_4",
            input_ref_label_4_shape=(256, 256, 30),
            output_name="output",
            bin_num=30,
            # bin_num=3,
            alpha=1.0,
        )

        # b) compile
        # model = model_helper.compile_model(model)
        from tensorflow.keras.losses import CategoricalCrossentropy
        from tensorflow.keras.metrics import CategoricalAccuracy
        from tensorflow.keras.optimizers import Adam

        model.compile(
            optimizer=Adam(lr=1e-4),
            loss=[CategoricalCrossentropy()],
            loss_weights=[1.0],
            metrics=[CategoricalAccuracy(name="acc")],
        )

        model.summary()

        # continue setting (weights)
        # if continue_from_model is not None:
        #     model.load_weights(os.path.join(save_weights_folder, continue_from_model))

    # 2. Dataset
    # ----------
    # training_batch_size: int = 4
    # val_batch_size: int = 4
    # test_batch_size: int = 4
    training_batch_size: int = 8
    val_batch_size: int = 8
    test_batch_size: int = 8

    # GS Dataset
    # main_image_file_names = tf.data.Dataset.list_files(
    #     training_main_image_folder + "/*", shuffle=True, seed=42
    # ).map(lambda path_name: tf.strings.split(path_name, sep="/")[-1])

    # training_input_dataset = (
    #     main_image_file_names.map(
    #         lambda fname: (
    #             # os.path.join(training_main_image_folder, fname),
    #             # os.path.join(training_ref_image_folder, fname),
    #             # os.path.join(training_ref_result_label_folder, fname),
    #             training_main_image_folder + "/" + fname,
    #             training_ref_image_folder + "/" + fname,
    #             training_ref_result_label_folder + "/" + fname,
    #         )
    #     )
    #     .map(
    #         lambda main_img_fname, ref_img_fname, ref_result_label_fname: (
    #             decode_png(main_img_fname),
    #             decode_png(ref_img_fname),
    #             decode_png(ref_result_label_fname, 3),
    #         ),
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE,
    #     )
    #     .map(
    #         lambda main_img, ref_img, ref_result_label: (
    #             tf_main_image_preprocessing_sequence(main_img),
    #             tf_ref_image_preprocessing_sequence(ref_img),
    #             tf_input_ref_label_1_preprocessing_function(ref_result_label),
    #         ),
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE,
    #     )
    # )
    # training_main_image_path_names = main_image_file_names.map(
    #     lambda fname: training_main_image_folder + "/" + fname,
    # )
    training_main_image_path_names = tf.data.Dataset.list_files(
        training_main_image_folder + "/*", shuffle=True, seed=42
    )
    training_main_images = training_main_image_path_names.map(
        decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    training_main_processed_images = training_main_images.map(
        tf_main_image_preprocessing_sequence,
    )

    training_input_dataset = training_main_processed_images

    # Main image
    # training_main_image_dataset = tf.data.Dataset.list_files(
    #     training_main_image_folder + "/*",
    #     shuffle=True,
    #     seed=42
    #     # training_main_image_folder,
    #     # shuffle=True,
    #     # seed=42,
    # )
    # training_main_image_dataset = training_main_image_dataset.map(
    #     decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE
    # ).map(tf_main_image_preprocessing_sequence)

    # # Ref image
    training_ref_image_dataset = tf.data.Dataset.list_files(
        training_ref_image_folder + "/*",
        shuffle=True,
        seed=42
        # training_ref_image_folder,
        # shuffle=True,
        # seed=42,
    )
    training_ref_image_dataset = training_ref_image_dataset.map(
        decode_png, num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).map(tf_ref_image_preprocessing_sequence)

    # # Ref Label
    training_ref_result_label_dataset = tf.data.Dataset.list_files(
        training_ref_result_label_folder + "/*",
        shuffle=True,
        seed=42
        # training_ref_result_label_folder,
        # shuffle=True,
        # seed=42,
    )
    training_ref_result_label_dataset = training_ref_result_label_dataset.map(
        lambda _img: decode_png(_img, 3),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    training_ref_result_label_1_dataset = training_ref_result_label_dataset.map(
        tf_input_ref_label_1_preprocessing_function
    )
    # # training_ref_result_label_2_dataset = training_ref_result_label_dataset.map(
    # #     tf_input_ref_label_2_preprocessing_function
    # # )
    # # training_ref_result_label_3_dataset = training_ref_result_label_dataset.map(
    # #     tf_input_ref_label_3_preprocessing_function
    # # )
    # # training_ref_result_label_4_dataset = training_ref_result_label_dataset.map(
    # #     tf_input_ref_label_4_preprocessing_function
    # # )

    training_ref_result_label_dataset2 = tf.data.Dataset.list_files(
        training_ref_result_label_folder + "/*",
        shuffle=True,
        seed=42
        # training_ref_result_label_folder,
        # shuffle=True,
        # seed=42,
    )
    training_ref_result_label_dataset2 = training_ref_result_label_dataset2.map(
        lambda _img: decode_png(_img, 3),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    training_ref_result_label_5_dataset = training_ref_result_label_dataset2.map(
        tf_input_ref_label_4_preprocessing_function
    )

    # Output Label
    training_output_main_label_dataset = tf.data.Dataset.list_files(
        training_output_main_label_folder + "/*",
        shuffle=True,
        seed=42
        # training_output_main_label_folder,
        # shuffle=True,
        # seed=42,
    )
    # training_output_main_label_dataset = training_output_main_label_dataset.map(
    #     lambda _img: decode_image(_img, 3),
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # )
    # training_output_main_label_dataset = training_output_main_label_dataset.map(
    #     tf_output_label_processing, num_parallel_calls=tf.data.experimental.AUTOTUNE
    # )
    # training_output_main_label_final_dataset = tf.data.Dataset.zip(
    #     (training_output_main_label_dataset, training_ref_result_label_4_dataset)
    # )
    # training_output_main_label_final_dataset.map(
    #     tf_output_label_processing, num_parallel_calls=tf.data.experimental.AUTOTUNE
    # )

    # Dataset
    training_dataset = (
        tf.data.Dataset.zip(
            (
                (
                    training_input_dataset,
                    training_ref_image_dataset,
                    training_ref_result_label_1_dataset,
                ),
                training_ref_result_label_5_dataset,
            )
        )
        .batch(training_batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    # training_dataset = (
    #     tf.data.Dataset.zip(
    #         (
    #             (
    #                 training_main_image_dataset,
    #                 training_ref_image_dataset,
    #                 training_ref_result_label_1_dataset,
    #                 # training_ref_result_label_2_dataset,
    #                 # training_ref_result_label_3_dataset,
    #                 # training_ref_result_label_4_dataset,
    #             ),
    #             (training_ref_result_label_5_dataset),
    #             # training_output_main_label_final_dataset,
    #         )
    #     )
    #     .batch(training_batch_size, drop_remainder=True)
    #     .cache()
    #     # .prefetch(tf.data.experimental.AUTOTUNE)
    # )
    training_samples = len(training_dataset) * training_batch_size

    # training_samples = 6

    # print(len(list(training_dataset)))
    # print(list(training_dataset)[0])

    # print("list(training_dataset)[0][0][0].shape")
    # print(list(training_dataset)[0][0][0].shape)
    # print("list(training_dataset)[0][0][1].shape")
    # print(list(training_dataset)[0][0][1].shape)
    # print("list(training_dataset)[0][0][2].shape")
    # print(list(training_dataset)[0][0][2].shape)
    # print("list(training_dataset)[0][1].shape")
    # print(list(training_dataset)[0][1].shape)

    # print("list(training_dataset)[1][0][0].shape")
    # print(list(training_dataset)[1][0][0].shape)
    # print("list(training_dataset)[1][0][1].shape")
    # print(list(training_dataset)[1][0][1].shape)
    # print("list(training_dataset)[1][0][2].shape")
    # print(list(training_dataset)[1][0][2].shape)
    # print("list(training_dataset)[1][1].shape")
    # print(list(training_dataset)[1][1].shape)

    # # Main image
    # val_main_image_dataset = tf.data.Dataset.list_files(
    #     val_main_image_folder + "/*",
    #     shuffle=True,
    #     seed=42
    #     # val_main_image_folder,
    #     # shuffle=True,
    #     # seed=42,
    # )
    # val_main_image_dataset = val_main_image_dataset.map(
    #     decode_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    # ).map(tf_main_image_preprocessing_sequence)

    # # Ref image
    # val_ref_image_dataset = tf.data.Dataset.list_files(
    #     val_ref_image_folder + "/*",
    #     shuffle=True,
    #     seed=42
    #     # val_ref_image_folder, shuffle=True, seed=42,
    # )
    # val_ref_image_dataset = val_ref_image_dataset.map(
    #     decode_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    # ).map(tf_ref_image_preprocessing_sequence)

    # # Ref Label
    # val_ref_result_label_dataset = tf.data.Dataset.list_files(
    #     val_ref_result_label_folder + "/*",
    #     shuffle=True,
    #     seed=42
    #     # val_ref_result_label_folder, shuffle=True, seed=42,
    # )
    # val_ref_result_label_dataset = val_ref_result_label_dataset.map(
    #     lambda _img: decode_image(_img, 3),
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # )

    # val_ref_result_label_1_dataset = val_ref_result_label_dataset.map(
    #     tf_input_ref_label_1_preprocessing_function
    # )
    # val_ref_result_label_2_dataset = val_ref_result_label_dataset.map(
    #     tf_input_ref_label_2_preprocessing_function
    # )
    # val_ref_result_label_3_dataset = val_ref_result_label_dataset.map(
    #     tf_input_ref_label_3_preprocessing_function
    # )
    # val_ref_result_label_4_dataset = val_ref_result_label_dataset.map(
    #     tf_input_ref_label_4_preprocessing_function
    # )
    # val_ref_result_label_5_dataset = val_ref_result_label_dataset.map(
    #     tf_input_ref_label_4_preprocessing_function
    # )

    # # Output Label
    # val_output_main_label_dataset = tf.data.Dataset.list_files(
    #     val_output_main_label_folder + "/*",
    #     shuffle=True,
    #     seed=42
    #     # val_output_main_label_folder, shuffle=True, seed=42,
    # )
    # val_output_main_label_dataset = val_output_main_label_dataset.map(
    #     lambda _img: decode_image(_img, 3),
    #     num_parallel_calls=tf.data.experimental.AUTOTUNE,
    # ).map(tf_output_label_processing)

    # # Dataset
    # val_dataset = (
    #     tf.data.Dataset.zip(
    #         (
    #             (
    #                 val_main_image_dataset,
    #                 val_ref_image_dataset,
    #                 val_ref_result_label_1_dataset,
    #                 val_ref_result_label_2_dataset,
    #                 val_ref_result_label_3_dataset,
    #                 val_ref_result_label_4_dataset,
    #             ),
    #             val_ref_result_label_5_dataset,
    #             # val_output_main_label_dataset,
    #         )
    #     )
    #     .batch(val_batch_size, drop_remainder=True)
    #     .cache()
    #     .prefetch(tf.data.experimental.AUTOTUNE)
    # )
    # val_samples = len(val_dataset) * val_batch_size

    # 2.1 Training, Validation Generator ---------
    # training_images_file_names = tf.data.Dataset.list_files(
    #     training_main_image_folder + "/*"
    # )

    # training_images_file_names = files_in_folder(training_main_image_folder)
    # training_samples = len(training_images_file_names)
    # training_sequence = RefTrackingSequence(
    #     image_file_names=training_images_file_names,
    #     main_image_folder_name=training_main_image_folder,
    #     ref_image_folder_name=training_ref_image_folder,
    #     ref_result_label_folder_name=training_ref_result_label_folder,
    #     output_label_folder_name=training_output_main_label_folder,
    #     batch_size=training_batch_size,
    #     shuffle=True,
    # )

    # val_images_file_names = files_in_folder(val_main_image_folder)
    # val_samples = len(val_images_file_names)
    # val_sequence = RefTrackingSequence(
    #     image_file_names=val_images_file_names,
    #     main_image_folder_name=val_main_image_folder,
    #     ref_image_folder_name=val_ref_image_folder,
    #     ref_result_label_folder_name=val_ref_result_label_folder,
    #     output_label_folder_name=val_output_main_label_folder,
    #     batch_size=val_batch_size,
    #     shuffle=False,
    # )

    # 3. Training
    # -----------
    # 3.1 Parameters ---------
    training_num_of_epochs: int = 200
    training_steps_per_epoch: int = training_samples // training_batch_size

    # val_freq: int = 1
    # val_steps: int = val_samples // val_batch_size

    # 3.2 Callbacks ---------
    apply_callbacks_after: int = 0
    # early_stopping_patience: int = training_num_of_epochs // (10 * val_freq)
    val_metric = model.compiled_metrics._metrics[-1].name
    val_checkpoint_metric = "val_" + val_metric
    model_checkpoint: Callback = ModelCheckpointAfter(
        os.path.join(
            save_weights_folder,
            check_weight_name
            + ".epoch_{epoch:02d}-val_loss_{val_loss:.3f}-"
            + val_checkpoint_metric
            + "_{"
            + val_checkpoint_metric
            + ":.3f}",
        ),
        verbose=1,
        after_epoch=apply_callbacks_after,
    )
    # early_stopping: Callback = EarlyStoppingAfter(
    #     patience=early_stopping_patience, verbose=1, after_epoch=apply_callbacks_after,
    # )
    tensorboard_cb: Callback = TensorBoard(log_dir=run_log_dir)
    callback_list: List[Callback] = [
        tensorboard_cb,
        model_checkpoint,
        # early_stopping,
    ]

    # 3.3 Training ---------
    # continue setting (initial epoch)
    initial_epoch = 0
    # if continue_initial_epoch is not None:
    #     initial_epoch = continue_initial_epoch

    history: History = model.fit(
        training_dataset,
        # training_sequence,
        epochs=training_num_of_epochs,
        verbose=1,
        callbacks=callback_list,
        # validation_data=val_dataset,
        # validation_data=val_sequence,
        shuffle=True,
        initial_epoch=initial_epoch,
        steps_per_epoch=training_steps_per_epoch,
        # validation_steps=val_steps,
        # validation_freq=val_freq,
        max_queue_size=10,
        workers=8,
        use_multiprocessing=True,
    )
