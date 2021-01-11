import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import tensorflow as tf
from image_keras.tf.utils.images import decode_png
from keras.utils import plot_model
from losses.losses_weighted_cce import WeightedCrossentropy
from ref_local_tracking.processings.tf.preprocessing import (
    tf_color_to_random_map,
    tf_input_ref_label_1_preprocessing_function,
    tf_input_ref_label_2_preprocessing_function,
    tf_input_ref_label_3_preprocessing_function,
    tf_input_ref_label_4_preprocessing_function,
    tf_main_image_preprocessing_sequence,
    tf_ref_image_preprocessing_sequence,
)
from utils.gc_storage import upload_blob
from utils.gc_tpu import tpu_initialize
from utils.run_setting import get_run_id
from utils.tf_images import decode_png

from _run.run_common_tpu import create_tpu, delete_tpu


def fill_bin(bin, bin_all_index, fill_empty_bin_with):
    fill_empty_with = tf.repeat(
        [fill_empty_bin_with], repeats=tf.shape(bin_all_index)[-1], axis=0
    )
    filled_bin = tf.concat([bin, [fill_empty_with]], axis=1)
    filled_bin = filled_bin[:, : tf.shape(bin_all_index)[-1], :]
    result = tf.squeeze(
        tf.gather(filled_bin, tf.cast(bin_all_index, tf.int32), axis=1), axis=0
    )
    return result


if __name__ == "__main__":
    # 1. Variables --------
    # 모델 이름
    model_name: str = "ref_local_tracking_model_003"
    # 테스트 배치 크기
    predict_test_batch_size: int = 1
    # 빈 크기
    bin_size: int = 30

    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Ref Local Predictor on TPU"
    )
    parser.add_argument(
        "--model_weight_path",
        required=True,
        type=str,
        help="Model to be predicted. \
            Full path of TF model which is accessable on cloud bucket. \
                ex) 'gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974'",
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
        "--predict_dataset_folder",
        type=str,
        default="tracking_test",
        help="Test dataset folder in google bucket. \
            ex) 'test_folder_name'",
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
    var_predict_dataset_folder: str = args.predict_dataset_folder
    run_id: str = args.run_id or get_run_id()
    run_id = run_id.replace(" ", "_")
    predict_id: str = "_predict__model_{}__run_{}".format(model_name, run_id)
    model_weight_path: str = args.model_weight_path
    without_tpu: bool = args.without_tpu

    # 2. Setup --------
    # tpu create
    if not without_tpu:
        create_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)

    # 2-2) Google bucket folder setting for dataset, tf_log, weights
    # data folder
    base_data_folder: str = os.path.join(gs_path, "data")
    predict_result_folder: str = os.path.join(base_data_folder, predict_id, "images")

    # dataset folder
    base_dataset_folder: str = os.path.join(gs_path, "dataset")
    predict_dataset_folder: str = os.path.join(
        base_dataset_folder, var_predict_dataset_folder
    )
    # sample files
    predict_sample_file_folder: str = os.path.join(
        predict_dataset_folder, "framed_sample"
    )
    # input - main image
    predict_main_image_folder: str = os.path.join(
        predict_dataset_folder, "framed_image", "zero"
    )
    # input - ref image
    predict_ref_image_folder: str = os.path.join(
        predict_dataset_folder, "framed_image", "p1"
    )
    # input - ref result label
    predict_ref_result_label_folder: str = os.path.join(
        predict_dataset_folder, "framed_label", "p1"
    )

    # 2-3) Setup results
    info: str = """
# Information ---------------------------
Predict ID: {}
Predict Dataset: {}
Predict Data Folder: {}/{}
-----------------------------------------
""".format(
        predict_id, predict_dataset_folder, base_data_folder, predict_id
    )
    print(info)
    tmp_info = "/tmp/info.txt"
    f = open(tmp_info, "w")
    f.write(info)
    f.close()
    upload_blob(
        bucket_name,
        tmp_info,
        os.path.join("data", predict_id, os.path.basename(tmp_info)),
    )

    # 3. Model compile --------
    def get_model():
        # model = tf.keras.models.load_model(model_weight_path)
        model = tf.keras.models.load_model(
            model_weight_path,
            custom_objects={
                "WeightedCrossentropy": WeightedCrossentropy(
                    weights=[
                        10.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]
                )
            },
        )
        # model.compile(
        #     optimizer=Adam(lr=1e-4),
        #     loss=[CategoricalCrossentropy()],
        #     loss_weights=[1.0],
        #     metrics=[CategoricalAccuracy(name="accuracy")],
        # )
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
            os.path.join("data", predict_id, os.path.basename(tmp_plot_model_img_path)),
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
    # 4-1) Predict dataset
    @tf.autograph.experimental.do_not_convert
    def spl(name):
        return tf.strings.split(name, sep="/")[-1]

    @tf.autograph.experimental.do_not_convert
    def s(a, b):
        return a + "/" + b

    predict_main_image_file_names = tf.data.Dataset.list_files(
        predict_sample_file_folder + "/*",
        shuffle=False
        # predict_main_image_folder + "/*", shuffle=False
    ).map(spl)

    predict_dataset = (
        predict_main_image_file_names.map(
            lambda fname: (
                s(predict_main_image_folder, fname),
                s(predict_ref_image_folder, fname),
                s(predict_ref_result_label_folder, fname),
                s(predict_ref_result_label_folder, fname),
                s(predict_ref_result_label_folder, fname),
                s(predict_ref_result_label_folder, fname),
                fname,
            )
        )
        .map(
            lambda main_img, ref_img, ref_label_1, ref_label_2, ref_label_3, ref_label_4, fname: (
                decode_png(main_img),
                decode_png(ref_img),
                decode_png(ref_label_1, 3),
                decode_png(ref_label_2, 3),
                decode_png(ref_label_3, 3),
                decode_png(ref_label_4, 3),
                fname,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda main_img, ref_img, ref_label_1, ref_label_2, ref_label_3, ref_label_4, fname: (
                (main_img, ref_img, ref_label_1, ref_label_2, ref_label_3, ref_label_4),
                tf_color_to_random_map(ref_label_4, ref_label_4, bin_size, 1),
                fname,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda input_imgs, color_info, fname: (
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
                color_info,
                fname,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )

    predict_dataset = predict_dataset.batch(
        predict_test_batch_size, drop_remainder=True
    ).prefetch(tf.data.experimental.AUTOTUNE)
    predict_samples = len(predict_dataset) * predict_test_batch_size

    # 5. Predict --------
    for predict_data in predict_dataset:
        predicted = model.predict(
            predict_data[0],
            batch_size=predict_test_batch_size,
            verbose=1,
            max_queue_size=1,
        )
        filled_bins = fill_bin(
            predict_data[1][1],
            bin_all_index=predict_data[1][0],
            fill_empty_bin_with=[255.0, 255.0, 255.0],
        )
        predicted_result = tf.squeeze(
            tf.gather(filled_bins, tf.argmax(predicted, axis=-1), axis=1)
        )

        print(s(predict_result_folder, predict_data[2])[0])
        encoded_predicted_image = tf.image.encode_png(
            tf.cast(predicted_result, tf.uint8)
        )
        tf.io.write_file(
            s(predict_result_folder, predict_data[2])[0], encoded_predicted_image
        )

    # 6. TPU shutdown --------
    if not without_tpu:
        delete_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)
