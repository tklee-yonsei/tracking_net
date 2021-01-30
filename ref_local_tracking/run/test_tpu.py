import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser
from typing import List, Tuple

import tensorflow as tf
from _run.run_common_tpu import create_tpu, delete_tpu, loss_coords
from keras.utils import plot_model
from ref_local_tracking.configs.losses import RefLoss
from ref_local_tracking.configs.metrics import RefMetric
from ref_local_tracking.configs.optimizers import RefOptimizer
from ref_local_tracking.run.dataset import (
    get_ref_tracking_dataset_for_cell_dataset,
    make_preprocessed_tf_dataset,
)
from utils.gc_storage import upload_blob
from utils.gc_tpu import tpu_initialize
from utils.run_setting import get_run_id

if __name__ == "__main__":
    # 1. Variables --------
    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Ref Local Tester on TPU"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Should be one in `ref_local_tracking/models` folder and method. \
            ex) 'ref_local_tracking_model_007'",
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
        help="Model to be tested. Full path of TF model which is accessable on cloud bucket. ex) 'gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974'",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        help="(Without space) Run with custom id. Be careful not to use duplicate IDs. If not specified, time is used as ID. ex) 'leetaekyu_210108_185302'",
    )
    parser.add_argument(
        "--ctpu_zone", type=str, help="VM, TPU zone. ex) 'us-central1-b'",
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
        help="Google Storage bucket name. ex) 'gs://bucket_name'",
    )
    parser.add_argument(
        "--test_dataset_folder",
        type=str,
        help="Test dataset folder in google bucket. ex) 'test_folder_name'",
    )
    parser.add_argument(
        "--without_tpu",
        action="store_true",
        help="With this option, TPU will not be used.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Optimizer. One of 'adam1' \
            ex) 'adam1'",
    )
    parser.add_argument(
        "--losses",
        type=loss_coords,
        action="append",
        help="Loss and weight pair. "
        "Loss should be exist in `ref_local_tracking.configs.losses`. "
        "- Case 1. 1 output  : `--losses 'categorical_crossentropy',1.0`"
        "- Case 2. 2 outputs : `--losses 'categorical_crossentropy',0.8 --losses 'weighted_cce',0.2`",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        action="append",
        help="Metrics. "
        "Metric should be exist in `ref_local_tracking.configs.metrics`."
        "- Case 1. 1 output, 1 metric  : `--metrics 'categorical_accuracy'`"
        "- Case 2. 1 output, 2 metric  : `--metrics 'categorical_accuracy' 'categorical_accuracy'`"
        "- Case 3. 2 output, (1, 1) metric  : `--metrics 'categorical_accuracy' --metrics 'categorical_accuracy'`"
        "- Case 4. 2 output, (1, 0) metric  : `--metrics categorical_accuracy --metrics none`"
        "- Case 5. 2 output, (2, 0) metric  : `--metrics categorical_accuracy categorical_accuracy --metrics none`",
    )
    args = parser.parse_args()

    # 1-2) Get variables
    # required
    model_name: str = args.model_name
    tpu_name: str = args.tpu_name
    # optional
    bin_size: int = args.bin_size or 30
    batch_size: int = args.batch_size or 8
    ctpu_zone: str = args.ctpu_zone or "us-central1-b"
    gs_path: str = args.gs_bucket_name or "gs://cell_dataset"
    var_test_dataset_folder: str = args.test_dataset_folder or "tracking_test"
    run_id: str = args.run_id or get_run_id()
    model_weight_path: str = args.model_weight_path
    without_tpu: bool = args.without_tpu
    # optimizer, losses, metrics
    optimizer: str = args.optimizer or RefOptimizer.get_default()
    loss_weight_tuple_list: List[Tuple[str, float]] = args.losses or [
        (RefLoss.get_default(), 1.0)
    ]
    metrics_list: List[List[str]] = args.metrics or [[RefMetric.get_default()]]
    # processing
    test_batch_size: int = batch_size
    bucket_name: str = gs_path.replace("gs://", "")
    run_id = run_id.replace(" ", "_")
    test_id: str = "test__model_{}__run_{}".format(model_name, run_id)

    # 2. Setup --------
    # tpu create
    if not without_tpu:
        create_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)

    # 2-2) Google bucket folder setting for dataset, tf_log, weights
    # test dataset
    test_dataset_folder: str = os.path.join(gs_path, "dataset", var_test_dataset_folder)
    test_inout_datasets = get_ref_tracking_dataset_for_cell_dataset(test_dataset_folder)

    # data folder
    base_data_folder: str = os.path.join(gs_path, "data")
    test_result_folder: str = os.path.join(base_data_folder, test_id)
    test_result_folder_without_gs: str = os.path.join("data", test_id)

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

        model_optimizer = RefOptimizer(optimizer).get_optimizer()
        model_loss_list = list(
            map(lambda el: RefLoss(el[0]).get_loss(), loss_weight_tuple_list)
        )
        model_loss_weight_list = list(map(lambda el: el[1], loss_weight_tuple_list))
        model_metrics_list = [
            list(
                filter(
                    lambda v: v, [RefMetric(metric).get_metric() for metric in metrics]
                )
            )
            for metrics in metrics_list
        ]
        output_keys = model.output_names
        if len(loss_weight_tuple_list) != len(output_keys):
            raise ValueError(
                "Number of `--losses` option(s) should be {}.".format(len(output_keys))
            )
        if len(metrics_list) != len(output_keys):
            raise ValueError(
                "Number of `--metrics` option(s) should be {}.".format(len(output_keys))
            )
        model_loss_dict = dict(zip(output_keys, model_loss_list))
        model_loss_weight_dict = dict(zip(output_keys, model_loss_weight_list))
        model_metrics_dict = dict(zip(output_keys, model_metrics_list))
        model.compile(
            optimizer=model_optimizer,
            loss=model_loss_dict,
            loss_weights=model_loss_weight_dict,
            metrics=model_metrics_dict,
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
                test_result_folder_without_gs,
                os.path.basename(tmp_plot_model_img_path),
            ),
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
    test_dataset = make_preprocessed_tf_dataset(
        batch_size=test_batch_size,
        inout_folder_tuple=test_inout_datasets,
        bin_size=bin_size,
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
