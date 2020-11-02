import os
import sys

sys.path.append(os.getcwd())

import time
from typing import List, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
from image_keras.custom.callbacks_after_epoch import (
    EarlyStoppingAfter,
    ModelCheckpointAfter,
)
from tensorflow.keras.callbacks import Callback, History, TensorBoard


def create_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(256, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(256, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )


def get_dataset(batch_size, is_training=True):
    split = "train" if is_training else "test"
    dataset, info = tfds.load(
        name="mnist", split=split, with_info=True, as_supervised=True, try_gcs=True
    )

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0

        return image, label

    dataset = dataset.map(scale)

    # Only shuffle and repeat the dataset in training. The advantage to have a
    # infinite dataset for training is to avoid the potential last partial batch
    # in each epoch, so users don't need to think about scaling the gradients
    # based on the actual batch size.
    if is_training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == "__main__":
    # 0. Prepare
    # ----------
    # training_id: 사용한 모델, Training 날짜
    # 0.1 ID ---------
    model_name: str = "mnist"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    config_id: str = "001"
    training_id: str = "_training__model_{}__config_{}__run_{}".format(
        model_name, config_id, run_id
    )
    print("# Information ---------------------------")
    print("Training ID: {}".format(training_id))
    # print("Training Dataset: {}".format(variable_training_dataset_folder))
    # print("Validation Dataset: {}".format(variable_validation_dataset_folder))
    # print("Config ID: {}".format(variable_config_id))
    print("-----------------------------------------")

    gs_path = "gs://cell_tracking_dataset"
    base_save_folder: str = os.path.join(gs_path, "save")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    tf_log_folder: str = os.path.join(base_save_folder, "tf_logs")
    run_log_dir: str = os.path.join(tf_log_folder, training_id)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver("tracking-1")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))
    strategy = tf.distribute.TPUStrategy(resolver)

    with strategy.scope():
        model = create_model()
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["sparse_categorical_accuracy"],
        )

    batch_size = 200
    steps_per_epoch = 60000 // batch_size
    validation_steps = 10000 // batch_size

    train_dataset = get_dataset(batch_size, is_training=True)
    test_dataset = get_dataset(batch_size, is_training=False)

    # 3.2 Callbacks ---------
    apply_callbacks_after: int = 0
    early_stopping_patience: int = 10

    val_metric = "sparse_categorical_accuracy"
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
    tensorboard_cb: Callback = TensorBoard(log_dir=run_log_dir, write_images=True)
    callback_list: List[Callback] = [tensorboard_cb, model_checkpoint, early_stopping]

    model.fit(
        train_dataset,
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        validation_steps=validation_steps,
    )

