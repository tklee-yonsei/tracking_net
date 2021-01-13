import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf
from image_keras.tf.keras.metrics.binary_class_mean_iou import binary_class_mean_iou
from image_keras.tf.utils.images import decode_png
from tensorflow.keras.models import Model
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def tf_equalize_histogram(tf_img):
    """
    Tensorflow Image Histogram Equalization

    https://stackoverflow.com/questions/42835247/how-to-implement-histogram-equalization-for-images-in-tensorflow

    Parameters
    ----------
    tf_img : `Tensor` of image
        Input `Tensor` image `tf_img`

    Returns
    -------
    `Tensor` of image
        Equalized histogram image of input `Tensor` image `tf_img`.
    """
    values_range = tf.constant([0.0, 255.0], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf.cast(tf_img, tf.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]
    img_shape = tf.shape(tf_img)

    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(
        tf.cast(cdf - cdf_min, tf.float32) * 255.0 / tf.cast(pix_cnt - 1, tf.float32)
    )
    px_map = tf.cast(px_map, tf.uint8)
    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(tf_img, tf.int32)), 2)
    return eq_hist


def tf_main_image_preprocessing_sequence(img):
    img = tf.image.resize(img, (256, 256), method=ResizeMethod.NEAREST_NEIGHBOR)
    img = tf_equalize_histogram(img)
    img = tf.cast(img, tf.float32)
    img = tf.math.divide(img, 255.0)
    img = tf.reshape(img, (256, 256, 1))
    return img


def l4_fmap_model(unet_model):
    return Model(
        inputs=unet_model.input,
        outputs=unet_model.get_layer(unet_model.layers[11].name).output,
    )


def l3_fmap_model(unet_model):
    return Model(
        inputs=unet_model.input,
        outputs=unet_model.get_layer(unet_model.layers[8].name).output,
    )


def l2_fmap_model(unet_model):
    return Model(
        inputs=unet_model.input,
        outputs=unet_model.get_layer(unet_model.layers[5].name).output,
    )


def l1_fmap_model(unet_model):
    return Model(
        inputs=unet_model.input,
        outputs=unet_model.get_layer(unet_model.layers[2].name).output,
    )


@tf.autograph.experimental.do_not_convert
def spl(name):
    return tf.strings.split(name, sep="/")[-1]


@tf.autograph.experimental.do_not_convert
def s(a, b):
    return a + "/" + b


if __name__ == "__main__":
    unet_model = tf.keras.models.load_model(
        "../test_resources/sample_unet_model",
        custom_objects={"binary_class_mean_iou": binary_class_mean_iou},
    )

    model = l4_fmap_model(unet_model)
    model.summary()

    predict_test_batch_size = 1
    predict_main_image_file_names = tf.data.Dataset.list_files(
        # "../test_resources/tracking_test/framed_image/p1/*", shuffle=False
        "../test_resources/tracking_test/framed_image/zero/*",
        shuffle=False,
    ).map(spl)
    predict_dataset = (
        predict_main_image_file_names.map(
            lambda fname: (
                # s("../test_resources/tracking_test/framed_image/p1", fname),
                s("../test_resources/tracking_test/framed_image/zero", fname),
                fname,
            )
        )
        .map(
            lambda input_path_name, fname: (decode_png(input_path_name), fname),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            lambda input_img, fname: (
                tf_main_image_preprocessing_sequence(input_img),
                fname,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )

    predict_dataset = (
        predict_dataset.batch(predict_test_batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # 5. Predict --------
    for predict_data in predict_dataset:
        predicted_batch_data = model.predict(
            predict_data[0],
            batch_size=predict_test_batch_size,
            verbose=1,
            max_queue_size=1,
        )
        for predicted_data in predicted_batch_data:
            proto = tf.make_tensor_proto(predicted_data)
            res = tf.make_ndarray(proto)
            # filename = tf.strings.split(predict_data[1], ".")[0] + "_l4_p1"
            filename = tf.strings.split(predict_data[1], ".")[0] + "_l4_zero"
            filename = s(".", filename)[0]
            print(filename)

            # tf.io.write_file(filename, proto)
            np.save(bytes.decode(filename.numpy()), res)
