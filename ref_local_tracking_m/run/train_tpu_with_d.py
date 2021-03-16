import os
import sys

sys.path.append(os.getcwd())

from argparse import ArgumentParser
from typing import List, Optional, Tuple

import tensorflow as tf
from _run.run_common_tpu import (
    check_all_exists_or_not,
    create_tpu,
    delete_tpu,
    loss_coords,
    setup_continuous_training,
)
from image_keras.tf.keras.metrics.binary_class_mean_iou import binary_class_mean_iou
from keras.utils import plot_model
from ref_local_tracking_m.configs.losses import RefLoss
from ref_local_tracking_m.configs.metrics import RefMetric
from ref_local_tracking_m.configs.optimizers import RefOptimizer
from ref_local_tracking_m.models.backbone.unet_l4_detach_activation import (
    unet_l4_detach_activation,
)
from ref_local_tracking_m.run.dataset import (
    get_ref_tracking_dataset_for_cell_dataset,
    make_preprocessed_tf_dataset,
    plot_and_upload_dataset,
)
from tensorflow.keras.callbacks import Callback, History, TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.gc_storage import upload_blob
from utils.gc_tpu import tpu_initialize
from utils.modules import load_module
from utils.run_setting import get_run_id

if __name__ == "__main__":
    # 1. Variables --------
    # 1-1) Variables with Parser
    parser: ArgumentParser = ArgumentParser(
        description="Arguments for Ref Local Trainer on TPU"
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
        help="Batch size for training and batch. Defaults to 8. \
            ex) 8",
    )
    parser.add_argument(
        "--training_epochs",
        type=int,
        help="Number of epochs for training. Defaults to 200. \
            ex) 200",
    )
    parser.add_argument(
        "--val_freq",
        type=int,
        help="Validate every this value. Defaults to 1. (validate every epoch) \
            ex) 1",
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
            Be careful not to use duplicate IDs. If not specified, timestamp will be ID. \
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
        "--training_dataset_folder",
        type=str,
        help="Training dataset folder in google bucket. \
            ex) 'training_folder_name'",
    )
    parser.add_argument(
        "--validation_dataset_folder",
        type=str,
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
        "--plot_sample",
        action="store_true",
        help="With this option, it will plot sample images.",
    )
    parser.add_argument(
        "--with_shared_unet",
        action="store_true",
        help="With this option, model uses shared U-Net.",
    )
    parser.add_argument(
        "--without_early_stopping",
        action="store_true",
        help="With this option, training will not early stop.",
    )
    parser.add_argument(
        "--freeze_unet_model",
        action="store_true",
        help="With this option, U-Net model would be freeze for training.",
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
        "Loss should be exist in `ref_local_tracking_m.configs.losses`. "
        "- Case 1. 1 output  : `--losses 'categorical_crossentropy',1.0`"
        "- Case 2. 2 outputs : `--losses 'categorical_crossentropy',0.8 --losses 'weighted_cce',0.2`",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        action="append",
        help="Metrics. "
        "Metric should be exist in `ref_local_tracking_m.configs.metrics`."
        "- Case 1. 1 output, 1 metric  : `--metrics 'categorical_accuracy'`"
        "- Case 2. 1 output, 2 metric  : `--metrics 'categorical_accuracy' 'categorical_accuracy'`"
        "- Case 3. 2 output, (1, 1) metric  : `--metrics 'categorical_accuracy' --metrics 'categorical_accuracy'`"
        "- Case 4. 2 output, (1, 0) metric  : `--metrics categorical_accuracy --metrics none`"
        "- Case 5. 2 output, (2, 0) metric  : `--metrics categorical_accuracy categorical_accuracy --metrics none`",
    )
    args = parser.parse_args()

    # 1-2) Set variables
    # required
    model_name: str = args.model_name
    tpu_name: str = args.tpu_name
    # optional
    bin_size: int = args.bin_size or 30
    val_freq: int = args.val_freq or 1
    batch_size: int = args.batch_size or 8
    training_epochs: int = args.training_epochs or 200
    ctpu_zone: str = args.ctpu_zone or "us-central1-b"
    gs_path: str = args.gs_bucket_name or "gs://cell_dataset"
    var_training_dataset_folder: str = args.training_dataset_folder or "tracking_training"
    var_validation_dataset_folder: str = args.validation_dataset_folder or "tracking_validation"
    pretrained_unet_path: Optional[str] = args.pretrained_unet_path
    freeze_unet_model: bool = args.freeze_unet_model
    without_early_stopping: bool = args.without_early_stopping
    with_shared_unet: bool = args.with_shared_unet
    plot_sample: bool = args.plot_sample
    run_id: str = args.run_id or get_run_id()
    # optimizer, losses, metrics
    optimizer: str = args.optimizer or RefOptimizer.get_default()
    loss_weight_tuple_list: List[Tuple[str, float]] = args.losses or [
        (RefLoss.get_default(), 1.0)
    ]
    metrics_list: List[List[str]] = args.metrics or [[RefMetric.get_default()]]
    # processing
    training_batch_size: int = batch_size
    val_batch_size: int = batch_size
    bucket_name: str = gs_path.replace("gs://", "")
    run_id = run_id.replace(" ", "_")
    training_id: str = "training__model_{}__run_{}".format(model_name, run_id)

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
    # 2-1) Create TPU
    create_tpu(tpu_name=tpu_name, ctpu_zone=ctpu_zone)

    # 2-2) TPU & Storage setting
    resolver = tpu_initialize(tpu_address=tpu_name, tpu_zone=ctpu_zone)
    strategy = tf.distribute.TPUStrategy(resolver)

    # 2-3) Google bucket folder setting for dataset, tf_log, weights
    # training dataset
    training_dataset_folder: str = os.path.join(
        gs_path, "dataset", var_training_dataset_folder
    )
    training_inout_datasets = get_ref_tracking_dataset_for_cell_dataset(
        training_dataset_folder
    )

    # validation dataset
    val_dataset_folder: str = os.path.join(
        gs_path, "dataset", var_validation_dataset_folder
    )
    val_inout_datasets = get_ref_tracking_dataset_for_cell_dataset(val_dataset_folder)

    # data folder
    base_data_folder: str = os.path.join(gs_path, "data")
    training_result_folder: str = os.path.join(base_data_folder, training_id)
    training_result_folder_without_gs: str = os.path.join("data", training_id)

    # save folder
    base_save_folder: str = os.path.join(gs_path, "save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    tf_log_folder: str = os.path.join(base_save_folder, "tf_logs")
    run_log_dir: str = os.path.join(tf_log_folder, training_id)

    # 2-4) Setup results
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
        os.path.join(training_result_folder_without_gs, os.path.basename(tmp_info)),
    )

    # 3. Model compile --------
    ref_tracking_model_module = load_module(
        module_name=model_name,
        file_path="ref_local_tracking_m/models/{}.py".format(model_name),
    )
    with strategy.scope():
        if pretrained_unet_path is None:
            unet_model = unet_l4_detach_activation(
                input_name="unet_input", output_name="unet_output"
            )
        else:
            unet_model = tf.keras.models.load_model(
                pretrained_unet_path,
                custom_objects={"binary_class_mean_iou": binary_class_mean_iou},
            )

        if not with_shared_unet:
            unet_model2 = tf.keras.models.clone_model(unet_model)
            unet_model2.set_weights(unet_model.get_weights())

            model = getattr(ref_tracking_model_module, model_name)(
                unet_l4_model_main=unet_model,
                unet_l4_model_ref=unet_model2,
                bin_num=bin_size,
            )
        else:
            model = getattr(ref_tracking_model_module, model_name)(
                unet_l4_model_main=unet_model,
                unet_l4_model_ref=unet_model,
                bin_num=bin_size,
            )

        # continue setting (weights)
        if continuous_model_name is not None:
            model = tf.keras.models.load_model(continuous_model_name)

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
                training_result_folder_without_gs,
                os.path.basename(tmp_plot_model_img_path),
            ),
        )
        model.summary()

    # 4. Dataset --------
    # 4-1) Training dataset
    training_dataset = make_preprocessed_tf_dataset(
        batch_size=training_batch_size,
        inout_folder_tuple=training_inout_datasets,
        bin_size=bin_size,
    )
    training_samples = len(training_dataset) * training_batch_size
    if plot_sample:
        plot_and_upload_dataset(
            dataset=training_dataset,
            batch_size=training_batch_size,
            bin_size=bin_size,
            bucket_name=bucket_name,
            upload_gs_folder=training_result_folder_without_gs,
        )

    # 4-2) Validation dataset
    val_dataset = make_preprocessed_tf_dataset(
        batch_size=val_batch_size,
        inout_folder_tuple=val_inout_datasets,
        bin_size=bin_size,
    )
    val_samples = len(val_dataset) * val_batch_size

    # 5. Training --------
    # 5-1) Parameters
    training_steps_per_epoch: int = training_samples // training_batch_size
    val_steps: int = val_samples // val_batch_size

    # callbacks
    model_checkpoint: Callback = ModelCheckpoint(
        os.path.join(save_weights_folder, training_id + ".epoch_{epoch:02d}"),
        verbose=1,
    )
    early_stopping_patience: int = training_epochs // (10 * val_freq)
    early_stopping: Callback = EarlyStopping(
        patience=early_stopping_patience, verbose=1
    )
    tensorboard_cb: Callback = TensorBoard(log_dir=run_log_dir)
    callback_list: List[Callback] = [tensorboard_cb, model_checkpoint]
    if not without_early_stopping:
        callback_list.append(early_stopping)

    # continue setting (initial epoch)
    initial_epoch = 0
    if continuous_epoch is not None:
        initial_epoch = continuous_epoch

    # 5-2) Training
    history: History = model.fit(
        training_dataset,
        epochs=training_epochs,
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
