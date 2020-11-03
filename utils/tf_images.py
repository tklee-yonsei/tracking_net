from typing import Tuple

import tensorflow as tf


def decode_image(filename: str, channels: int = 1):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_png(bits, channels)
    return image


def tf_img_to_minmax(
    img, threshold, min_max: Tuple[int, int] = (0, 1),
):
    cond = tf.greater(img, tf.ones_like(img) * threshold)
    mask = tf.where(
        cond, tf.ones_like(img) * min_max[1], tf.ones_like(img) * min_max[0]
    )
    return mask


def tf_equalize_histogram(image):
    """
    Tensorflow Image Histogram Equalization

    https://stackoverflow.com/questions/42835247/how-to-implement-histogram-equalization-for-images-in-tensorflow

    Parameters
    ----------
    image : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    values_range = tf.constant([0.0, 255.0], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf.cast(image, tf.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]
    img_shape = tf.shape(image)

    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(
        tf.cast(cdf - cdf_min, tf.float32) * 255.0 / tf.cast(pix_cnt - 1, tf.float32)
    )
    px_map = tf.cast(px_map, tf.uint8)
    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist
