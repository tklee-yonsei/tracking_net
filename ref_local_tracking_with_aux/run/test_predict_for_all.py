import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from _run.run_common_tpu import create_tpu, delete_tpu
from image_keras.tf.keras.metrics.binary_class_mean_iou import binary_class_mean_iou
from keras.utils import plot_model
from ref_local_tracking_with_aux.models.test_ref_local_tracking_model_025_aux import (
    test_ref_local_tracking_model_025_aux,
)
from ref_local_tracking_with_aux.run.dataset import (
    combine_folder_file,
    get_ref_tracking_dataset_for_cell_dataset,
    get_ref_tracking_test_sample_predict_dataset,
    make_preprocessed_tf_dataset,
    make_preprocessed_tf_test_dataset,
    make_preprocessed_tf_test_sample_dataset,
)
from utils.gc_storage import upload_blob
from utils.gc_tpu import tpu_initialize
from utils.run_setting import get_run_id

# python3 ref_local_tracking_with_aux/run/test_predict.py \
# --run_id "leetaekyu_20210216_103352" \
# --model_name "test_ref_local_tracking_model_025_aux" \
# --model_weight_path "gs://cell_dataset/save/weights/training__model_ref_local_tracking_model_025_aux__run_leetaekyu_20210203_231220.epoch_45" \
# --bin_size 30 \
# --batch_size 1 \
# --tpu_name "none" \
# --without_tpu


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
    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Ref Local Predictor on TPU"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Should be one in `ref_local_tracking_with_aux/models` folder and method. \
            ex) 'ref_local_tracking_with_aux_model_007'",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        help="Color bin size. Defaults to 30. \
            ex) 30",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for test. Defaults to 8. \
            ex) 8",
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
        help="Google Storage bucket name. \
            ex) 'gs://bucket_name'",
    )
    parser.add_argument(
        "--predict_dataset_folder",
        type=str,
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
    # required
    model_name: str = args.model_name
    tpu_name: str = args.tpu_name
    # optional
    bin_size: int = args.bin_size or 30
    batch_size: int = args.batch_size or 1
    ctpu_zone: str = args.ctpu_zone or "us-central1-b"
    gs_path: str = args.gs_bucket_name or "gs://cell_dataset"
    var_predict_dataset_folder: str = args.predict_dataset_folder or "tracking_test"
    run_id: str = args.run_id or get_run_id()
    model_weight_path: str = args.model_weight_path
    without_tpu: bool = args.without_tpu
    # processing
    predict_test_batch_size: int = batch_size
    bucket_name: str = gs_path.replace("gs://", "")
    run_id = run_id.replace(" ", "_")
    predict_testset_id: str = "predict_testset__model_{}__run_{}".format(
        model_name, run_id
    )

    # 2. Setup --------
    # tpu create
    if not without_tpu:
        create_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)

    # 2-2) Google bucket folder setting for dataset, tf_log, weights
    # dataset
    predict_dataset_folder: str = os.path.join(
        gs_path, "dataset", var_predict_dataset_folder
    )
    predict_inout_datasets = get_ref_tracking_dataset_for_cell_dataset(
        predict_dataset_folder
    )

    # data folder
    base_data_folder: str = os.path.join(gs_path, "data")
    predict_testset_result_folder: str = os.path.join(
        base_data_folder, predict_testset_id
    )
    predict_testset_result_folder_without_gs: str = os.path.join(
        "data", predict_testset_id
    )
    predict_image_result_folder: str = os.path.join(
        base_data_folder, predict_testset_id, "images"
    )

    # 2-3) Setup results
    info: str = """
# Information ---------------------------
Predict ID: {}
Predict Dataset: {}
Predict Data Folder: {}/{}
-----------------------------------------
""".format(
        predict_testset_id, predict_dataset_folder, base_data_folder, predict_testset_id
    )
    print(info)
    tmp_info = "/tmp/info.txt"
    f = open(tmp_info, "w")
    f.write(info)
    f.close()
    upload_blob(
        bucket_name,
        tmp_info,
        os.path.join("data", predict_testset_id, os.path.basename(tmp_info)),
    )

    # 3. Model compile --------
    def get_model():
        unet_model = tf.keras.models.load_model(
            "gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974",
            custom_objects={"binary_class_mean_iou": binary_class_mean_iou},
        )
        model = test_ref_local_tracking_model_025_aux(
            unet_l4_model_main=unet_model, unet_l4_model_ref=unet_model
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
                "data", predict_testset_id, os.path.basename(tmp_plot_model_img_path)
            ),
        )
        model.summary()
        return model

    if not without_tpu:
        # 2-1) TPU & Storage setting
        resolver = tpu_initialize(tpu_address=tpu_name, tpu_zone=ctpu_zone)
        strategy = tf.distribute.TPUStrategy(resolver)

        with strategy.scope():
            model = get_model()
    else:
        model = get_model()

    # 4. Dataset --------
    # 4-1) Predict dataset
    predict_test_dataset = make_preprocessed_tf_test_dataset(
        batch_size=predict_test_batch_size,
        inout_folder_tuple=predict_inout_datasets,
        bin_size=bin_size,
    )
    predict_test_samples = len(predict_test_dataset) * predict_test_batch_size

    # 5. Predict --------
    for predict_data in predict_test_dataset:
        image_filename = predict_data[2][0].numpy().decode()
        filename = image_filename[: image_filename.rfind(".")]

        print("Processing {} ...".format(filename))

        np.save(
            "/home/tklee/results/{}_input_prev_32.npy".format(filename),
            predict_data[0][2],
        )
        np.save(
            "/home/tklee/results/{}_input_prev_64.npy".format(filename),
            predict_data[0][3],
        )
        np.save(
            "/home/tklee/results/{}_input_prev_128.npy".format(filename),
            predict_data[0][4],
        )
        np.save(
            "/home/tklee/results/{}_input_prev_256.npy".format(filename),
            predict_data[0][5],
        )
        np.save(
            "/home/tklee/results/{}_output_current_32.npy".format(filename),
            predict_data[3][1],
        )
        np.save(
            "/home/tklee/results/{}_output_current_64.npy".format(filename),
            predict_data[3][2],
        )
        np.save(
            "/home/tklee/results/{}_output_current_128.npy".format(filename),
            predict_data[3][3],
        )
        np.save(
            "/home/tklee/results/{}_output_current_256.npy".format(filename),
            predict_data[3][4],
        )
        predicted = model.predict(
            predict_data[0],
            batch_size=predict_test_batch_size,
            verbose=1,
            max_queue_size=1,
        )
        np.save("/home/tklee/results/{}_32.npy".format(filename), predicted[1][0])
        np.save("/home/tklee/results/{}_64.npy".format(filename), predicted[2][0])
        np.save("/home/tklee/results/{}_128.npy".format(filename), predicted[3][0])
        np.save("/home/tklee/results/{}_256.npy".format(filename), predicted[4][0])

    #     # filled_bins = fill_bin(
    #     #     predict_data[1][1],
    #     #     bin_all_index=predict_data[1][0],
    #     #     fill_empty_bin_with=[255.0, 255.0, 255.0],
    #     # )
    #     # predicted_result = tf.squeeze(
    #     #     tf.gather(filled_bins, tf.argmax(predicted[0], axis=-1), axis=1)
    #     # )

    #     # print(combine_folder_file(predict_image_result_folder, predict_data[2])[0])
    #     # encoded_predicted_image = tf.image.encode_png(
    #     #     tf.cast(predicted_result, tf.uint8)
    #     # )
    #     # tf.io.write_file(
    #     #     combine_folder_file(predict_image_result_folder, predict_data[2])[0],
    #     #     encoded_predicted_image,
    #     # )

    # 6. TPU shutdown --------
    if not without_tpu:
        delete_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)
