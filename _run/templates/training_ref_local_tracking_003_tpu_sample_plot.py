import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser
from typing import List, Optional

import tensorflow as tf
from _run.run_common_tpu import (
    check_all_exists_or_not,
    create_tpu,
    delete_tpu,
    setup_continuous_training,
)
from image_keras.tf.keras.metrics.binary_class_mean_iou import binary_class_mean_iou
from image_keras.tf.utils.images import decode_png
from keras.utils import plot_model
from ref_local_tracking.models.backbone.unet_l4 import unet_l4
from ref_local_tracking.models.ref_local_tracking_model_003 import (
    ref_local_tracking_model_003,
)
from ref_local_tracking.processings.tf.preprocessing import (
    tf_color_to_random_map,
    tf_input_ref_label_1_preprocessing_function,
    tf_input_ref_label_2_preprocessing_function,
    tf_input_ref_label_3_preprocessing_function,
    tf_input_ref_label_4_preprocessing_function,
    tf_main_image_preprocessing_sequence,
    tf_output_label_processing,
    tf_ref_image_preprocessing_sequence,
)
from tensorflow.keras.callbacks import Callback, History, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.gc_storage import upload_blob
from utils.gc_tpu import tpu_initialize
from utils.plot_dataset import plot_samples, take_from_dataset_at_all_batch
from utils.run_setting import get_run_id
from utils.tf_images import decode_png

if __name__ == "__main__":
    # 1. Variables --------
    # 모델 이름
    model_name: str = "ref_local_tracking_model_003"
    # 트레이닝 배치 크기
    training_batch_size: int = 8
    # 검증 배치 크기
    val_batch_size: int = 8
    # 트레이닝 에포크 수
    training_num_of_epochs: int = 200
    # 검증을 매번 `val_freq` 에포크마다
    val_freq: int = 1
    # 빈 크기
    bin_size: int = 30

    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Ref Local Trainer on TPU"
    )
    parser.add_argument(
        "--continuous_model_name",
        type=str,
        help="Training will be continue for this `model`. \
            Full path of TF model which is accessable on cloud bucket. \
                ex) 'gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974'",
    )
    parser.add_argument(
        "--continuous_epoch",
        type=int,
        help="Training will be continue from this `epoch`. \
            If model trained during 12 epochs, this will be 12. \
                ex) 12",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="(Without space) Run with custom id. \
            Be careful not to use duplicate IDs. If not specified, time is used as ID. \
                ex) 'leetaekyu_210108_185302'",
    )
    parser.add_argument(
        "--ctpu_zone",
        type=str,
        default="us-central1-b",
        help="VM, TPU zone. \
            ex) 'us-central1-b'",
    )
    parser.add_argument(
        "--tpu_name",
        type=str,
        required=True,
        help="TPU name. \
            ex) 'leetaekyu-1-trainer'",
    )
    parser.add_argument(
        "--gs_bucket_name",
        type=str,
        default="gs://cell_dataset",
        help="Google Storage bucket name. \
            ex) 'gs://bucket_name'",
    )
    parser.add_argument(
        "--training_dataset_folder",
        type=str,
        default="tracking_training",
        help="Training dataset folder in google bucket. \
            ex) 'training_folder_name'",
    )
    parser.add_argument(
        "--validation_dataset_folder",
        type=str,
        default="tracking_validation",
        help="Validation dataset folder in google bucket. \
            ex) 'val_folder_name'",
    )
    parser.add_argument(
        "--pretrained_unet_path",
        type=str,
        help="Pretrained U-Net L4 model in google bucket. It must contains 'gs://'. \
            ex) 'gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974'",
    )
    parser.add_argument(
        "--freeze_unet_model",
        action="store_true",
        help="With this option, U-Net model would be freeze for training.",
    )
    parser.add_argument(
        "--plot_sample",
        action="store_true",
        help="With this option, it will plot sample images.",
    )
    args = parser.parse_args()

    # 1-2) Get variables
    tpu_name: str = args.tpu_name
    ctpu_zone: str = args.ctpu_zone
    bucket_name: str = args.gs_bucket_name.replace("gs://", "")
    gs_path: str = args.gs_bucket_name
    var_training_dataset_folder: str = args.training_dataset_folder
    var_validation_dataset_folder: str = args.validation_dataset_folder
    run_id: str = args.run_id or get_run_id()
    run_id = run_id.replace(" ", "_")
    training_id: str = "_training__model_{}__run_{}".format(model_name, run_id)
    pretrained_unet_path: Optional[str] = args.pretrained_unet_path
    freeze_unet_model: bool = args.freeze_unet_model
    plot_sample: bool = args.plot_sample

    # 1-3) continuous
    continuous_model_name: Optional[str] = args.continuous_model_name
    continuous_epoch: Optional[int] = args.continuous_epoch
    # continuous parameter check
    if not check_all_exists_or_not([continuous_model_name, continuous_epoch]):
        raise RuntimeError(
            "`continuous_model_name` and `continuous_epoch` should both exists or not."
        )
    training_id = (
        setup_continuous_training(continuous_model_name, continuous_epoch)
        or training_id
    )

    # 2. Setup --------
    # tpu create
    create_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)

    # 2-1) TPU & Storage setting
    resolver = tpu_initialize(tpu_address=tpu_name)
    strategy = tf.distribute.TPUStrategy(resolver)

    # 2-2) Google bucket folder setting for dataset, tf_log, weights
    # save folder
    base_save_folder: str = os.path.join(gs_path, "save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    tf_log_folder: str = os.path.join(base_save_folder, "tf_logs")

    # data folder
    base_data_folder: str = os.path.join(gs_path, "data")
    training_result_folder: str = os.path.join(base_data_folder, training_id)

    # dataset folder
    base_dataset_folder: str = os.path.join(gs_path, "dataset")
    training_dataset_folder: str = os.path.join(
        base_dataset_folder, var_training_dataset_folder
    )
    val_dataset_folder: str = os.path.join(
        base_dataset_folder, var_validation_dataset_folder
    )
    # input - main image
    training_main_image_folder: str = os.path.join(
        training_dataset_folder, "framed_image", "zero"
    )
    val_main_image_folder: str = os.path.join(
        val_dataset_folder, "framed_image", "zero"
    )
    # input - ref image
    training_ref_image_folder: str = os.path.join(
        training_dataset_folder, "framed_image", "p1"
    )
    val_ref_image_folder: str = os.path.join(val_dataset_folder, "framed_image", "p1")
    # input - ref result label
    training_ref_result_label_folder: str = os.path.join(
        training_dataset_folder, "framed_label", "p1"
    )
    val_ref_result_label_folder: str = os.path.join(
        val_dataset_folder, "framed_label", "p1"
    )
    # output - main label
    training_output_main_label_folder: str = os.path.join(
        training_dataset_folder, "framed_label", "zero"
    )
    val_output_main_label_folder: str = os.path.join(
        val_dataset_folder, "framed_label", "zero"
    )

    # callback folder
    check_weight_name = training_id[1:]
    run_log_dir: str = os.path.join(tf_log_folder, training_id)

    # 2-3) Setup results
    info: str = """
# Information ---------------------------
Training ID: {}
Training Dataset: {}
Validation Dataset: {}
Tensorboard Log Folder: {}
Training Data Folder: {}/{}
-----------------------------------------
""".format(
        training_id,
        training_dataset_folder,
        val_dataset_folder,
        run_log_dir,
        base_data_folder,
        training_id,
    )
    print(info)
    tmp_info = "/tmp/info.txt"
    f = open(tmp_info, "w")
    f.write(info)
    f.close()
    upload_blob(
        bucket_name,
        tmp_info,
        os.path.join("data", training_id, os.path.basename(tmp_info)),
    )

    # 3. Model compile --------
    with strategy.scope():
        if pretrained_unet_path is None:
            unet_model = unet_l4(input_name="unet_input", output_name="unet_output")
        else:
            unet_model = tf.keras.models.load_model(
                pretrained_unet_path,
                custom_objects={"binary_class_mean_iou": binary_class_mean_iou},
            )

        model = ref_local_tracking_model_003(
            pre_trained_unet_l4_model=unet_model,
            input_main_image_shape=(256, 256, 1),
            input_ref_image_shape=(256, 256, 1),
            input_ref_label_1_shape=(32, 32, 30),
            input_ref_label_2_shape=(64, 64, 30),
            input_ref_label_3_shape=(128, 128, 30),
            input_ref_label_4_shape=(256, 256, 30),
            bin_num=30,
            unet_trainable=(not freeze_unet_model),
        )

        # continue setting (weights)
        if continuous_model_name is not None:
            model = tf.keras.models.load_model(continuous_model_name)

        model.compile(
            optimizer=Adam(lr=1e-4),
            loss=[CategoricalCrossentropy()],
            loss_weights=[1.0],
            metrics=[CategoricalAccuracy(name="accuracy")],
        )
        tmp_plot_model_img_path = "/tmp/model.png"
        plot_model(
            model,
            show_shapes=True,
            to_file=tmp_plot_model_img_path,
            expand_nested=True,
            dpi=144,
        )
        upload_blob(
            bucket_name,
            tmp_plot_model_img_path,
            os.path.join(
                "data", training_id, os.path.basename(tmp_plot_model_img_path)
            ),
        )
        model.summary()

    # 4. Dataset --------
    # 4-1) Training dataset
    @tf.autograph.experimental.do_not_convert
    def spl(name):
        return tf.strings.split(name, sep="/")[-1]

    @tf.autograph.experimental.do_not_convert
    def s(a, b):
        return a + "/" + b

    training_main_image_file_names = tf.data.Dataset.list_files(
        training_main_image_folder + "/*", shuffle=True, seed=42
    ).map(spl)
    training_dataset = (
        training_main_image_file_names.map(
            lambda fname: (
                (
                    s(training_main_image_folder, fname),
                    s(training_ref_image_folder, fname),
                    s(training_ref_result_label_folder, fname),
                    s(training_ref_result_label_folder, fname),
                    s(training_ref_result_label_folder, fname),
                    s(training_ref_result_label_folder, fname),
                ),
                (training_output_main_label_folder + "/" + fname),
            )
        )
        .map(
            lambda input_path_names, output_label_fname: (
                (
                    decode_png(input_path_names[0]),
                    decode_png(input_path_names[1]),
                    decode_png(input_path_names[2], 3),
                    decode_png(input_path_names[3], 3),
                    decode_png(input_path_names[4], 3),
                    decode_png(input_path_names[5], 3),
                ),
                (decode_png(output_label_fname, 3)),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda input_imgs, output_label: (
                input_imgs,
                tf_color_to_random_map(input_imgs[5], output_label[0], bin_size, 1),
                output_label,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda input_imgs, color_info, output_label: (
                (
                    tf_main_image_preprocessing_sequence(input_imgs[0]),
                    tf_ref_image_preprocessing_sequence(input_imgs[1]),
                    tf_input_ref_label_1_preprocessing_function(
                        input_imgs[2], color_info, bin_size
                    ),
                    tf_input_ref_label_2_preprocessing_function(
                        input_imgs[3], color_info, bin_size
                    ),
                    tf_input_ref_label_3_preprocessing_function(
                        input_imgs[4], color_info, bin_size
                    ),
                    tf_input_ref_label_4_preprocessing_function(
                        input_imgs[5], color_info, bin_size
                    ),
                ),
                (tf_output_label_processing(output_label, color_info, bin_size)),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )
    training_dataset = (
        training_dataset.batch(training_batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    training_samples = len(training_dataset) * training_batch_size

    # Training dataset sample ploting
    if plot_sample:

        def ratio_img_to_img(img):
            img = img * 255
            return tf.cast(img, tf.uint8)

        def ratio_img_to_np_img(img):
            return ratio_img_to_img(img).numpy()

        def bin_img_to_np_arr_img(img, bin_num):
            imgs = ratio_img_to_img(img)
            bin_imgs = []
            for bin_index in range(bin_num):
                bin_img = imgs[:, :, bin_index : bin_index + 1]
                bin_imgs.append(bin_img.numpy())
            return bin_imgs

        ratio_img_to_np_arr_img = lambda img: [ratio_img_to_np_img(img)]
        bin_img_to_np_arr_img_default_bin = lambda img: bin_img_to_np_arr_img(
            img, bin_size
        )

        input_images, output_images = take_from_dataset_at_all_batch(
            training_dataset,
            (
                [
                    ratio_img_to_np_arr_img,
                    ratio_img_to_np_arr_img,
                    bin_img_to_np_arr_img_default_bin,
                    bin_img_to_np_arr_img_default_bin,
                    bin_img_to_np_arr_img_default_bin,
                    bin_img_to_np_arr_img_default_bin,
                ],
                [bin_img_to_np_arr_img_default_bin],
            ),
        )

        for b_i in range(training_batch_size):
            print("Sample ploting {}/{}...".format(b_i + 1, training_batch_size))
            filename = "/tmp/sample_img_{}.png".format(b_i)
            plot_samples(input_images[b_i] + output_images[b_i], filename, 4, 4)
            upload_blob(
                bucket_name,
                filename,
                os.path.join("data", training_id, os.path.basename(filename)),
            )

    # 4-2) Validation dataset
    val_main_image_file_names = tf.data.Dataset.list_files(
        val_main_image_folder + "/*", shuffle=True, seed=42
    ).map(spl)
    val_dataset = (
        val_main_image_file_names.map(
            lambda fname: (
                (
                    s(val_main_image_folder, fname),
                    s(val_ref_image_folder, fname),
                    s(val_ref_result_label_folder, fname),
                    s(val_ref_result_label_folder, fname),
                    s(val_ref_result_label_folder, fname),
                    s(val_ref_result_label_folder, fname),
                ),
                (val_output_main_label_folder + "/" + fname),
            )
        )
        .map(
            lambda input_path_names, output_label_fname: (
                (
                    decode_png(input_path_names[0]),
                    decode_png(input_path_names[1]),
                    decode_png(input_path_names[2], 3),
                    decode_png(input_path_names[3], 3),
                    decode_png(input_path_names[4], 3),
                    decode_png(input_path_names[5], 3),
                ),
                (decode_png(output_label_fname, 3)),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda input_imgs, output_label: (
                input_imgs,
                tf_color_to_random_map(input_imgs[5], output_label[0], bin_size, 1),
                output_label,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda input_imgs, color_info, output_label: (
                (
                    tf_main_image_preprocessing_sequence(input_imgs[0]),
                    tf_ref_image_preprocessing_sequence(input_imgs[1]),
                    tf_input_ref_label_1_preprocessing_function(
                        input_imgs[2], color_info, bin_size
                    ),
                    tf_input_ref_label_2_preprocessing_function(
                        input_imgs[3], color_info, bin_size
                    ),
                    tf_input_ref_label_3_preprocessing_function(
                        input_imgs[4], color_info, bin_size
                    ),
                    tf_input_ref_label_4_preprocessing_function(
                        input_imgs[5], color_info, bin_size
                    ),
                ),
                (tf_output_label_processing(output_label, color_info, bin_size)),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )
    val_dataset = (
        val_dataset.batch(val_batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_samples = len(val_dataset) * val_batch_size

    # 5. Training --------
    # 5-1) Parameters
    training_steps_per_epoch: int = training_samples // training_batch_size
    val_steps: int = val_samples // val_batch_size

    val_metric = model.compiled_metrics._metrics[-1].name
    val_checkpoint_metric = "val_" + val_metric
    model_checkpoint: Callback = ModelCheckpoint(
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
    )
    early_stopping_patience: int = training_num_of_epochs // (10 * val_freq)
    early_stopping: Callback = EarlyStopping(
        patience=early_stopping_patience, verbose=1
    )
    tensorboard_cb: Callback = TensorBoard(log_dir=run_log_dir)
    callback_list: List[Callback] = [
        tensorboard_cb,
        model_checkpoint,
        early_stopping,
    ]

    # continue setting (initial epoch)
    initial_epoch = 0
    if continuous_model_name is not None:
        initial_epoch = continuous_epoch

    # 5-2) Training
    history: History = model.fit(
        training_dataset,
        epochs=training_num_of_epochs,
        verbose=1,
        callbacks=callback_list,
        validation_data=val_dataset,
        shuffle=True,
        initial_epoch=initial_epoch,
        steps_per_epoch=training_steps_per_epoch,
        validation_steps=val_steps,
        validation_freq=val_freq,
        max_queue_size=10,
        workers=8,
        use_multiprocessing=True,
    )

    # 6. TPU shutdown --------
    delete_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)
