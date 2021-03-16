import tensorflow as tf
import tf_clahe
from image_keras.tf.utils.images import (
    tf_change_order,
    tf_equalize_histogram,
    tf_generate_random_color_map,
    tf_image_detach_with_id_color_list,
    tf_shrink3D,
)
from tensorflow.python.ops.image_ops_impl import ResizeMethod


def tf_main_image_preprocessing_sequence(img):
    img = tf.image.resize(img, (256, 256), method=ResizeMethod.NEAREST_NEIGHBOR)
    img = tf_clahe.clahe(img, tile_grid_size=(8, 8), clip_limit=2.0)
    # img = tf_equalize_histogram(img)
    img = tf.cast(img, tf.float32)
    img = tf.math.divide(img, 255.0)
    img = tf.reshape(img, (256, 256, 1))
    return img


def tf_ref_image_preprocessing_sequence(img):
    img = tf.image.resize(img, (256, 256), method=ResizeMethod.NEAREST_NEIGHBOR)
    img = tf_clahe.clahe(img, tile_grid_size=(8, 8), clip_limit=2.0)
    # img = tf_equalize_histogram(img)
    img = tf.cast(img, tf.float32)
    img = tf.math.divide(img, 255.0)
    img = tf.reshape(img, (256, 256, 1))
    return img


def tf_color_to_random_map(ref_label_img, label_img, bin_size, exclude_first=1):
    ref_id_color_list = tf_generate_random_color_map(
        ref_label_img, bin_size=bin_size, shuffle_exclude_first=exclude_first, seed=42
    )
    return ref_id_color_list


def tf_image_detach_with_id_color_probability_list(
    color_img, id_color_list, bin_num: int, resize_by_power_of_two: int = 0,
):
    result = tf_image_detach_with_id_color_list(color_img, id_color_list, bin_num, 1.0)
    ratio = 2 ** resize_by_power_of_two
    result2 = tf_shrink3D(
        result, tf.shape(result)[-3] // ratio, tf.shape(result)[-2] // ratio, bin_num
    )
    result2 = tf.divide(result2, ratio ** 2)
    return result2


def tf_input_ref_label_preprocessing_function(label, color_info, bin_size=30):
    result = tf_image_detach_with_id_color_probability_list(
        label, color_info, bin_size, 0
    )
    result = tf.reshape(result, (256 // (2 ** 0), 256 // (2 ** 0), bin_size))
    result = tf_change_order(result, color_info[0])
    result = tf.squeeze(result)
    return result


def tf_output_label_processing(label, color_info, bin_size=30):
    result = tf_image_detach_with_id_color_probability_list(
        label, color_info, bin_size, 0
    )
    result = tf.reshape(result, (256 // (2 ** 0), 256 // (2 ** 0), bin_size))
    result = tf_change_order(result, color_info[0])
    return result


def tf_unet_output_label_processing(label):
    img = tf.image.resize(label, (256, 256), method=ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.cast(img, tf.float32)
    img = tf.math.divide(img, 255.0)
    img = tf.reshape(img, (256, 256, 1))
    return img

