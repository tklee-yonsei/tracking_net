import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import tensorflow as tf
from image_keras.tf.keras.metrics.binary_class_mean_iou import binary_class_mean_iou
from image_keras.tf.utils.images import decode_png
from keras.utils import plot_model
from ref_local_tracking.processings.tf.preprocessing import (
    tf_main_image_preprocessing_sequence,
    tf_unet_output_label_processing,
)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from utils.gc_storage import upload_blob
from utils.gc_tpu import tpu_initialize
from utils.run_setting import get_run_id
from utils.tf_images import decode_png

from _run.run_common_tpu import create_tpu, delete_tpu

if __name__ == "__main__":
    # 1. Variables --------
    # 모델 이름
    model_name: str = "unet_l4"
    # 테스트 배치 크기
    test_batch_size: int = 8

    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for U-Net Tester on TPU"
    )
    parser.add_argument(
        "--model_weight_path",
        required=True,
        type=str,
        help="Model to be tested. Full path of TF model which is accessable on cloud bucket. ex) 'gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974'",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="(Without space) Run with custom id. Be careful not to use duplicate IDs. If not specified, time is used as ID. ex) 'leetaekyu_210108_185302'",
    )
    parser.add_argument(
        "--ctpu_zone",
        type=str,
        default="us-central1-b",
        help="VM, TPU zone. ex) 'us-central1-b'",
    )
    parser.add_argument(
        "--tpu_name",
        type=str,
        required=True,
        help="TPU name. ex) 'leetaekyu-1-trainer'",
    )
    parser.add_argument(
        "--gs_bucket_name",
        type=str,
        default="gs://cell_dataset",
        help="Google Storage bucket name. ex) 'gs://bucket_name'",
    )
    parser.add_argument(
        "--test_dataset_folder",
        type=str,
        default="tracking_test",
        help="Test dataset folder in google bucket. ex) 'test_folder_name'",
    )
    args = parser.parse_args()

    # 1-2) Get variables
    tpu_name: str = args.tpu_name
    ctpu_zone: str = args.ctpu_zone
    bucket_name: str = args.gs_bucket_name.replace("gs://", "")
    gs_path: str = args.gs_bucket_name
    var_test_dataset_folder: str = args.test_dataset_folder
    run_id: str = args.run_id or get_run_id()
    run_id = run_id.replace(" ", "_")
    test_id: str = "_test__model_{}__run_{}".format(model_name, run_id)
    model_weight_path: str = args.model_weight_path

    # 2. Setup --------
    # tpu create
    create_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)

    # 2-1) TPU & Storage setting
    resolver = tpu_initialize(tpu_address=tpu_name)
    strategy = tf.distribute.TPUStrategy(resolver)

    # 2-2) Google bucket folder setting for dataset, tf_log, weights
    # data folder
    base_data_folder: str = os.path.join(gs_path, "data")
    test_result_folder: str = os.path.join(base_data_folder, test_id)

    # dataset folder
    base_dataset_folder: str = os.path.join(gs_path, "dataset")
    test_dataset_folder: str = os.path.join(
        base_dataset_folder, var_test_dataset_folder
    )
    # input - main image
    test_main_image_folder: str = os.path.join(
        test_dataset_folder, "framed_image", "zero"
    )
    # output - main label
    test_output_main_label_folder: str = os.path.join(test_dataset_folder, "bw_label")

    # callback folder
    # check_weight_name = test_id[1:]
    # run_log_dir: str = os.path.join(tf_log_folder, training_id)

    # 2-3) Setup results
    print("# Information ---------------------------")
    print("Test ID: {}".format(test_id))
    print("Test Dataset: {}".format(test_dataset_folder))
    print("Test Data Folder: {}/{}".format(base_data_folder, test_id))
    print("-----------------------------------------")

    # 3. Model compile --------
    with strategy.scope():
        model = tf.keras.models.load_model(
            model_weight_path,
            custom_objects={"binary_class_mean_iou": binary_class_mean_iou},
        )
        model.compile(
            optimizer=Adam(lr=1e-4),
            loss=[BinaryCrossentropy()],
            loss_weights=[1.0],
            metrics=[binary_class_mean_iou, BinaryAccuracy(name="accuracy")],
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
            os.path.join("data", test_id, os.path.basename(tmp_plot_model_img_path)),
        )
        model.summary()

    # 4. Dataset --------
    # 4-1) Test dataset
    @tf.autograph.experimental.do_not_convert
    def spl(name):
        return tf.strings.split(name, sep="/")[-1]

    @tf.autograph.experimental.do_not_convert
    def s(a, b):
        return a + "/" + b

    test_main_image_file_names = tf.data.Dataset.list_files(
        test_main_image_folder + "/*", shuffle=True, seed=42
    ).map(spl)
    test_dataset = (
        test_main_image_file_names.map(
            lambda fname: (
                s(test_main_image_folder, fname),
                test_output_main_label_folder + "/" + fname,
            )
        )
        .map(
            lambda input_path_name, output_label_fname: (
                decode_png(input_path_name),
                decode_png(output_label_fname),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda input_img, output_label: (
                tf_main_image_preprocessing_sequence(input_img),
                tf_unet_output_label_processing(output_label),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )
    test_dataset = (
        test_dataset.batch(test_batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    test_samples = len(test_dataset) * test_batch_size

    # 5. Test --------
    test_loss, test_mean_iou, test_acc = model.evaluate(
        test_dataset, workers=8, use_multiprocessing=True
    )

    print("test_loss: {}".format(test_loss))
    print("test_binary_class_mean_iou: {}".format(test_mean_iou))
    print("test_acc: {}".format(test_acc))

    # 6. TPU shutdown --------
    delete_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)
