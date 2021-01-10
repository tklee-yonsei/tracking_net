import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import tensorflow as tf
from image_keras.tf.utils.images import decode_png
from keras.utils import plot_model
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
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from utils.gc_storage import upload_blob
from utils.gc_tpu import tpu_initialize
from utils.run_setting import get_run_id
from utils.tf_images import decode_png

from _run.run_common_tpu import create_tpu, delete_tpu

if __name__ == "__main__":
    # 1. Variables --------
    # 모델 이름
    model_name: str = "ref_local_tracking_model_003"
    # 테스트 배치 크기
    test_batch_size: int = 8
    # 빈 크기
    bin_size: int = 30

    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Ref Local Tester on TPU"
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
    parser.add_argument(
        "--without_tpu",
        action="store_true",
        help="With this option, TPU will not be used.",
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
    without_tpu: bool = args.without_tpu

    # 2. Setup --------
    # tpu create
    if not without_tpu:
        create_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)

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
    # input - ref image
    test_ref_image_folder: str = os.path.join(test_dataset_folder, "framed_image", "p1")
    # input - ref result label
    test_ref_result_label_folder: str = os.path.join(
        test_dataset_folder, "framed_label", "p1"
    )
    # output - main label
    test_output_main_label_folder: str = os.path.join(
        test_dataset_folder, "framed_label", "zero"
    )

    # 2-3) Setup results
    info: str = """
# Information ---------------------------
Test ID: {}
Test Dataset: {}
Test Data Folder: {}/{}
-----------------------------------------
""".format(
        test_id, test_dataset_folder, base_data_folder, test_id
    )
    print(info)
    tmp_info = "/tmp/info.txt"
    f = open(tmp_info, "w")
    f.write(info)
    f.close()
    upload_blob(
        bucket_name,
        tmp_info,
        os.path.join("data", test_id, os.path.basename(tmp_info)),
    )

    # 3. Model compile --------
    def get_model():
        model = tf.keras.models.load_model(model_weight_path)
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
            os.path.join("data", test_id, os.path.basename(tmp_plot_model_img_path)),
        )
        model.summary()
        return model

    if not without_tpu:
        # 2-1) TPU & Storage setting
        resolver = tpu_initialize(tpu_address=tpu_name)
        strategy = tf.distribute.TPUStrategy(resolver)

        with strategy.scope():
            model = get_model()
    else:
        model = get_model()

    # 4. Dataset --------
    # 4-1) Test dataset
    @tf.autograph.experimental.do_not_convert
    def spl(name):
        return tf.strings.split(name, sep="/")[-1]

    @tf.autograph.experimental.do_not_convert
    def s(a, b):
        return a + "/" + b

    test_main_image_file_names = tf.data.Dataset.list_files(
        test_main_image_folder + "/*", shuffle=False
    ).map(spl)
    test_dataset = (
        test_main_image_file_names.map(
            lambda fname: (
                (
                    s(test_main_image_folder, fname),
                    s(test_ref_image_folder, fname),
                    s(test_ref_result_label_folder, fname),
                    s(test_ref_result_label_folder, fname),
                    s(test_ref_result_label_folder, fname),
                    s(test_ref_result_label_folder, fname),
                ),
                (test_output_main_label_folder + "/" + fname),
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
    test_dataset = (
        test_dataset.batch(test_batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    test_samples = len(test_dataset) * test_batch_size

    # 5. Test --------
    test_loss, test_acc = model.evaluate(
        test_dataset, workers=8, use_multiprocessing=True
    )

    result: str = """
# Result ---------------------------
Test Loss: {}
Test Accuracy: {}
-----------------------------------------
""".format(
        test_loss, test_acc
    )
    print(result)
    tmp_result = "/tmp/result.txt"
    f = open(tmp_result, "w")
    f.write(result)
    f.close()
    upload_blob(
        bucket_name,
        tmp_result,
        os.path.join("data", test_id, os.path.basename(tmp_result)),
    )

    # 6. TPU shutdown --------
    if not without_tpu:
        delete_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)
