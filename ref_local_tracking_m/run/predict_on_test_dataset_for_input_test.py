import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from _run.run_common_tpu import create_tpu, delete_tpu
from keras.utils import plot_model
from ref_local_tracking_m.models.backbone.unet_l4 import unet_l4
from ref_local_tracking_m.models.ref_local_tracking_model_021_m import (
    ref_local_tracking_model_021_m,
)
from ref_local_tracking_m.run.dataset import (
    get_ref_tracking_test_sample_predict_dataset,
    make_preprocessed_tf_test_sample_dataset,
)
from tensorflow.keras.models import Model
from utils.gc_storage import upload_blob
from utils.gc_tpu import tpu_initialize
from utils.plot_dataset import plot_samples
from utils.run_setting import get_run_id


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


def ratio_img_to_img(img):
    img = img * 255
    return tf.cast(img, tf.uint8)


def bin_img_to_np_arr_img(img, bin_num):
    imgs = ratio_img_to_img(img)
    bin_imgs = []
    for bin_index in range(bin_num):
        bin_img = imgs[:, :, :, bin_index : bin_index + 1]
        bin_imgs.append(bin_img.numpy())
    return bin_imgs


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
        help="Should be one in `ref_local_tracking_m/models` folder and method. \
            ex) 'ref_local_tracking_m_model_007'",
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
    predict_inout_datasets = get_ref_tracking_test_sample_predict_dataset(
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
        # U-Net
        unet_model = unet_l4(input_name="unet_input", output_name="unet_output")
        unet_model2 = tf.keras.models.clone_model(unet_model)
        unet_model2.set_weights(unet_model.get_weights())

        # ref local m
        ref_local_model = ref_local_tracking_model_021_m(unet_model, unet_model2)
        shrink_layer_names = [
            ref_local_model.layers[6].name,
            ref_local_model.layers[8].name,
            ref_local_model.layers[10].name,
            ref_local_model.layers[15].name,
        ]
        model = Model(
            inputs=ref_local_model.inputs,
            outputs=[
                ref_local_model.get_layer(shrink_layer_names[0]).output,
                ref_local_model.get_layer(shrink_layer_names[1]).output,
                ref_local_model.get_layer(shrink_layer_names[2]).output,
                ref_local_model.get_layer(shrink_layer_names[3]).output,
            ],
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
    predict_test_dataset = make_preprocessed_tf_test_sample_dataset(
        batch_size=predict_test_batch_size,
        inout_folder_tuple=predict_inout_datasets,
        bin_size=bin_size,
    )
    predict_test_samples = len(predict_test_dataset) * predict_test_batch_size

    # 5. Predict --------
    for predict_data in predict_test_dataset:
        predicted = model.predict(
            predict_data[0],
            batch_size=predict_test_batch_size,
            verbose=1,
            max_queue_size=1,
        )

        input_label = bin_img_to_np_arr_img(predict_data[0][2], bin_size)
        r0 = bin_img_to_np_arr_img(predicted[0], bin_size)
        r1 = bin_img_to_np_arr_img(predicted[1], bin_size)
        r2 = bin_img_to_np_arr_img(predicted[2], bin_size)
        r3 = bin_img_to_np_arr_img(predicted[3], bin_size)

        import pylab
        from matplotlib import pyplot as plt

        def plot_samples(images, save_file_name, element_width=3, element_height=3):
            r = 1
            c = len(images)

            f, axarr = plt.subplots(
                r, c, figsize=(element_width * c, element_height * r)
            )
            f.patch.set_facecolor("xkcd:white")
            for c_i in range(c):
                axarr[c_i].xaxis.set_ticks([])
                axarr[c_i].yaxis.set_ticks([])
                axarr[c_i].axis("off")

            pylab.gray()

            for row_index, image_row in enumerate(images):
                axarr[row_index].imshow(np.squeeze(image_row))

            f.savefig(save_file_name)

        for b_i in range(batch_size):
            print("Sample ploting {}/{}...".format(b_i + 1, batch_size))
            this_file_name = predict_data[2][0].numpy().decode()
            this_file_name = this_file_name[: this_file_name.rfind(".")]
            plot_samples(input_label, "{}_input_label.png".format(this_file_name), 4, 4)
            plot_samples(r0, "{}_r0.png".format(this_file_name), 4, 4)
            plot_samples(r1, "{}_r1.png".format(this_file_name), 4, 4)
            plot_samples(r2, "{}_r2.png".format(this_file_name), 4, 4)
            plot_samples(r3, "{}_r3.png".format(this_file_name), 4, 4)

    # 6. TPU shutdown --------
    if not without_tpu:
        delete_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)
