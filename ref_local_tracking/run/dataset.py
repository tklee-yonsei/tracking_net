import os
from typing import List, Tuple

import tensorflow as tf
from image_keras.tf.utils.images import decode_png
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
from utils.gc_storage import upload_blob
from utils.plot_dataset import plot_samples, take_from_dataset_at_all_batch


def get_ref_tracking_dataset_for_cell_dataset(
    base_folder: str,
) -> Tuple[Tuple[str, str, str], Tuple[str]]:
    """
    Make ref tracking dataset for `cell_dataset`.

    It consists of:
    - Input
        - Main image: "`base_folder`/framed_image/zero"
        - Ref image: "`base_folder`/framed_image/p1"
        - Ref label: "`base_folder`/framed_label/p1"
    - Output
        - Main label: "`base_folder`/framed_label/zero"

    Parameters
    ----------
    base_folder : str
        `cell_dataset` base folder. 

    Returns
    -------
    Tuple[Tuple[str, str, str], Tuple[str]]
        [description]
    """
    # [Input] main image
    input_main_image_folder: str = os.path.join(base_folder, "framed_image", "zero")
    # [Input] ref image
    input_ref_image_folder: str = os.path.join(base_folder, "framed_image", "p1")
    # [Input] ref result label
    input_ref_result_label_folder: str = os.path.join(base_folder, "framed_label", "p1")
    # [Output] main label
    output_main_label_folder: str = os.path.join(base_folder, "framed_label", "zero")

    return (
        [
            input_main_image_folder,
            input_ref_image_folder,
            input_ref_result_label_folder,
        ],
        [output_main_label_folder],
    )


@tf.autograph.experimental.do_not_convert
def get_filename_from_fullpath(name):
    return tf.strings.split(name, sep="/")[-1]


@tf.autograph.experimental.do_not_convert
def combine_folder_file(a, b):
    return a + "/" + b


def make_preprocessed_tf_dataset(
    batch_size: int,
    inout_folder_tuple: Tuple[Tuple[str, str, str], Tuple[str]],
    bin_size: int,
):
    input_main_image_folder = inout_folder_tuple[0][0]
    input_ref_image_folder = inout_folder_tuple[0][1]
    input_ref_label_folder = inout_folder_tuple[0][2]
    output_main_label_folder = inout_folder_tuple[1][0]

    main_image_file_names = tf.data.Dataset.list_files(
        input_main_image_folder + "/*", shuffle=True, seed=42
    ).map(get_filename_from_fullpath)
    dataset = (
        main_image_file_names.map(
            lambda fname: (
                (
                    combine_folder_file(input_main_image_folder, fname),
                    combine_folder_file(input_ref_image_folder, fname),
                    combine_folder_file(input_ref_label_folder, fname),
                    combine_folder_file(input_ref_label_folder, fname),
                    combine_folder_file(input_ref_label_folder, fname),
                    combine_folder_file(input_ref_label_folder, fname),
                ),
                (output_main_label_folder + "/" + fname),
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
    dataset = (
        dataset.batch(batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


def make_preprocessed_tf_dataset(
    batch_size: int,
    inout_folder_tuple: Tuple[Tuple[str, str, str], Tuple[str]],
    bin_size: int,
):
    input_main_image_folder = inout_folder_tuple[0][0]
    input_ref_image_folder = inout_folder_tuple[0][1]
    input_ref_label_folder = inout_folder_tuple[0][2]
    output_main_label_folder = inout_folder_tuple[1][0]

    main_image_file_names = tf.data.Dataset.list_files(
        input_main_image_folder + "/*", shuffle=True, seed=42
    ).map(get_filename_from_fullpath)
    dataset = (
        main_image_file_names.map(
            lambda fname: (
                (
                    combine_folder_file(input_main_image_folder, fname),
                    combine_folder_file(input_ref_image_folder, fname),
                    combine_folder_file(input_ref_label_folder, fname),
                    combine_folder_file(input_ref_label_folder, fname),
                    combine_folder_file(input_ref_label_folder, fname),
                    combine_folder_file(input_ref_label_folder, fname),
                ),
                (output_main_label_folder + "/" + fname),
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
    dataset = (
        dataset.batch(batch_size, drop_remainder=True)
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset


def plot_and_upload_dataset(
    dataset, batch_size: int, bin_size: int, bucket_name: str, upload_gs_folder: str
):
    def ratio_img_to_img(img):
        img = img * 255
        return tf.cast(img, tf.uint8)

    def ratio_img_to_np_img(img):
        return ratio_img_to_img(img).numpy()

    def bin_img_to_np_arr_img(img, bin_num):
        imgs = ratio_img_to_img(img)
        bin_imgs = []
        for bin_index in range(bin_num):
            bin_img = imgs[:, :, bin_index : bin_index + 1]
            bin_imgs.append(bin_img.numpy())
        return bin_imgs

    ratio_img_to_np_arr_img = lambda img: [ratio_img_to_np_img(img)]
    bin_img_to_np_arr_img_default_bin = lambda img: bin_img_to_np_arr_img(img, bin_size)

    input_images, output_images = take_from_dataset_at_all_batch(
        dataset,
        (
            [
                ratio_img_to_np_arr_img,
                ratio_img_to_np_arr_img,
                bin_img_to_np_arr_img_default_bin,
                bin_img_to_np_arr_img_default_bin,
                bin_img_to_np_arr_img_default_bin,
                bin_img_to_np_arr_img_default_bin,
            ],
            [bin_img_to_np_arr_img_default_bin],
        ),
    )

    for b_i in range(batch_size):
        print("Sample ploting {}/{}...".format(b_i + 1, batch_size))
        filename = "/tmp/sample_img_{}.png".format(b_i)
        plot_samples(input_images[b_i] + output_images[b_i], filename, 4, 4)
        upload_blob(
            bucket_name,
            filename,
            os.path.join(upload_gs_folder, os.path.basename(filename)),
        )
