import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import skimage.io as io
import cv2
import skimage.transform as trans
from keras.preprocessing.image import ImageDataGenerator

from seg_models.pre_process import gray_image_to_ratio
from utils.image_transform import (
    color_map_generate,
    get_rgb_color_cv2,
    image_detach_with_id_color_list,
    image_detach_with_id_color_probability_list,
)


def adjust_data(
    main_gray_img: np.ndarray,
    ref1_gray_img: np.ndarray,
    ref1_result: np.ndarray,
    label: np.ndarray,
    bin_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    그레이스케일 이미지와 라벨 이미지의 값 범위를 조정합니다.

    Parameters
    ----------
    main_gray_img : np.ndarray
        현재 이미지 (그레이스케일)
    ref1_gray_img : np.ndarray
        참조 이미지 (그레이스케일)
    ref1_result : np.ndarray
        참조 결과 라벨 이미지 (컬러)
    label : np.ndarray
        라벨 이미지 (컬러)
    bin_num : int
        빈 개수

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        그레이스케일 이미지의 범위를 0~255에서, 0~1로 바꾸고, 라벨의 경우는 0~1 범위로 바꾼 후, threshold에 따라, 0과 1의 이진 이미지로 만듭니다.
    """
    main_gray_img = gray_image_to_ratio(main_gray_img)
    ref1_gray_img = gray_image_to_ratio(ref1_gray_img)
    batch_size: int = ref1_result.shape[0]
    r_ref1_results = np.zeros(ref1_result.shape[0:3] + (bin_num,))
    r_labels = np.zeros(label.shape[:3] + (bin_num,))
    for i in range(0, batch_size):
        ref1_label_img = ref1_result[i, :]
        ref1_label_img = ref1_label_img.astype(np.uint8)
        prev_color_list: List[Tuple[int, int, int]] = get_rgb_color_cv2(ref1_label_img)
        prev_id_color_list: List[Tuple[int, Tuple[int, int, int]]] = color_map_generate(
            prev_color_list
        )
        r_ref1_result = image_detach_with_id_color_list(
            ref1_result[i, :], prev_id_color_list, bin_num
        )
        r_ref1_results[i, :] = r_ref1_result
        r_label = image_detach_with_id_color_list(
            label[i, :], prev_id_color_list, bin_num
        )
        r_labels[i, :] = r_label
    return main_gray_img, ref1_gray_img, r_ref1_results, r_labels


def adjust_data2(
    main_gray_img: np.ndarray,
    ref1_gray_img: np.ndarray,
    ref1_result: np.ndarray,
    label: np.ndarray,
    bin_num: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    그레이스케일 이미지와 라벨 이미지의 값 범위를 조정합니다.

    Parameters
    ----------
    main_gray_img : np.ndarray
        현재 이미지 (그레이스케일)
    ref1_gray_img : np.ndarray
        참조 이미지 (그레이스케일)
    ref1_result : np.ndarray
        참조 결과 라벨 이미지 (컬러)
    label : np.ndarray
        라벨 이미지 (컬러)
    bin_num : int
        빈 개수

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        그레이스케일 이미지의 범위를 0~255에서, 0~1로 바꾸고, 라벨의 경우는 0~1 범위로 바꾼 후, threshold에 따라, 0과 1의 이진 이미지로 만듭니다.
    """
    main_gray_img = gray_image_to_ratio(main_gray_img)
    ref1_gray_img = gray_image_to_ratio(ref1_gray_img)
    batch_size: int = ref1_result.shape[0]
    r_ref1_results = np.zeros(ref1_result.shape[0:3] + (bin_num,))
    r_labels = np.zeros(label.shape[:3] + (bin_num,))
    r_ref1_size_1_2 = tuple(int(el / 2) for el in ref1_result.shape[1:3])
    r_ref1_results_1_2 = np.zeros(
        (ref1_result.shape[0],) + r_ref1_size_1_2 + (bin_num,)
    )
    r_ref1_size_1_4 = tuple(int(el / 4) for el in ref1_result.shape[1:3])
    r_ref1_results_1_4 = np.zeros(
        (ref1_result.shape[0],) + r_ref1_size_1_4 + (bin_num,)
    )
    r_ref1_size_1_8 = tuple(int(el / 8) for el in ref1_result.shape[1:3])
    r_ref1_results_1_8 = np.zeros(
        (ref1_result.shape[0],) + r_ref1_size_1_8 + (bin_num,)
    )
    for i in range(0, batch_size):
        ref1_label_img = ref1_result[i, :]
        ref1_label_img = ref1_label_img.astype(np.uint8)
        prev_color_list: List[Tuple[int, int, int]] = get_rgb_color_cv2(ref1_label_img)
        prev_id_color_list: List[Tuple[int, Tuple[int, int, int]]] = color_map_generate(
            prev_color_list
        )

        ref1_result_img_1_2 = ref1_result[i, :]
        ref1_result_img_1_2 = cv2.resize(
            ref1_result_img_1_2, r_ref1_size_1_2, interpolation=cv2.INTER_NEAREST
        )
        r_ref1_result_1_2 = image_detach_with_id_color_list(
            ref1_result_img_1_2, prev_id_color_list, bin_num
        )
        r_ref1_results_1_2[i, :] = r_ref1_result_1_2

        ref1_result_img_1_4 = ref1_result[i, :]
        ref1_result_img_1_4 = cv2.resize(
            ref1_result_img_1_4, r_ref1_size_1_4, interpolation=cv2.INTER_NEAREST
        )
        r_ref1_result_1_4 = image_detach_with_id_color_list(
            ref1_result_img_1_4, prev_id_color_list, bin_num
        )
        r_ref1_results_1_4[i, :] = r_ref1_result_1_4

        ref1_result_img_1_8 = ref1_result[i, :]
        ref1_result_img_1_8 = cv2.resize(
            ref1_result_img_1_8, r_ref1_size_1_8, interpolation=cv2.INTER_NEAREST
        )
        r_ref1_result_1_8 = image_detach_with_id_color_list(
            ref1_result_img_1_8, prev_id_color_list, bin_num
        )
        r_ref1_results_1_8[i, :] = r_ref1_result_1_8

        r_ref1_result = image_detach_with_id_color_list(
            ref1_result[i, :], prev_id_color_list, bin_num
        )
        r_ref1_results[i, :] = r_ref1_result
        r_label = image_detach_with_id_color_list(
            label[i, :], prev_id_color_list, bin_num
        )
        r_labels[i, :] = r_label
    return (
        main_gray_img,
        ref1_gray_img,
        r_ref1_results,
        r_ref1_results_1_2,
        r_ref1_results_1_4,
        r_ref1_results_1_8,
        r_labels,
    )


def adjust_data3(
    main_gray_img: np.ndarray,
    ref1_gray_img: np.ndarray,
    ref1_result: np.ndarray,
    label: np.ndarray,
    bin_num: int,
    index: int = 0,
    save_to_dir_adjust: str = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    그레이스케일 이미지와 라벨 이미지의 값 범위를 조정합니다.

    Parameters
    ----------
    main_gray_img : np.ndarray
        현재 이미지 (그레이스케일)
    ref1_gray_img : np.ndarray
        참조 이미지 (그레이스케일)
    ref1_result : np.ndarray
        참조 결과 라벨 이미지 (컬러)
    label : np.ndarray
        라벨 이미지 (컬러)
    bin_num : int
        빈 개수

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        그레이스케일 이미지의 범위를 0~255에서, 0~1로 바꾸고, 라벨의 경우는 0~1 범위로 바꾼 후, threshold에 따라, 0과 1의 이진 이미지로 만듭니다.
    """
    batch_size: int = ref1_result.shape[0]
    main_gray_img = gray_image_to_ratio(main_gray_img)
    if save_to_dir_adjust is not None:
        for i in range(0, batch_size):
            cv2.imwrite(
                os.path.join(save_to_dir_adjust, "main_img_{}_{}.png".format(index, i)),
                main_gray_img[i, :] * 255,
            )

    ref1_gray_img = gray_image_to_ratio(ref1_gray_img)
    if save_to_dir_adjust is not None:
        for i in range(0, batch_size):
            cv2.imwrite(
                os.path.join(save_to_dir_adjust, "ref1_img_{}_{}.png".format(index, i)),
                ref1_gray_img[i, :] * 255,
            )

    batch_size: int = ref1_result.shape[0]
    r_ref1_results = np.zeros(ref1_result.shape[0:3] + (bin_num,))
    r_labels = np.zeros(label.shape[:3] + (bin_num,))
    r_ref1_size_1_2 = tuple(int(el / 2) for el in ref1_result.shape[1:3])
    r_ref1_results_1_2 = np.zeros(
        (ref1_result.shape[0],) + r_ref1_size_1_2 + (bin_num,)
    )
    r_ref1_size_1_4 = tuple(int(el / 4) for el in ref1_result.shape[1:3])
    r_ref1_results_1_4 = np.zeros(
        (ref1_result.shape[0],) + r_ref1_size_1_4 + (bin_num,)
    )
    r_ref1_size_1_8 = tuple(int(el / 8) for el in ref1_result.shape[1:3])
    r_ref1_results_1_8 = np.zeros(
        (ref1_result.shape[0],) + r_ref1_size_1_8 + (bin_num,)
    )
    for i in range(0, batch_size):
        ref1_label_img = ref1_result[i, :]
        ref1_label_img = ref1_label_img.astype(np.uint8)
        prev_color_list: List[Tuple[int, int, int]] = get_rgb_color_cv2(
            ref1_label_img, False
        )
        # print("prev_color_list: {}".format(prev_color_list))
        prev_id_color_list: List[Tuple[int, Tuple[int, int, int]]] = color_map_generate(
            prev_color_list
        )
        # print("prev_id_color_list: {}".format(prev_id_color_list))

        ref1_result_img_1_2 = ref1_result[i, :]
        r_ref1_result_1_2 = image_detach_with_id_color_probability_list(
            ref1_result_img_1_2, prev_id_color_list, bin_num, 1
        )
        r_ref1_results_1_2[i, :] = r_ref1_result_1_2
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "prev_results_1_2__{}_{}_{}.png".format(index, i, j),
                        ),
                        r_ref1_results_1_2[i, :, :, j] * 255,
                    )

        ref1_result_img_1_4 = ref1_result[i, :]
        r_ref1_result_1_4 = image_detach_with_id_color_probability_list(
            ref1_result_img_1_4, prev_id_color_list, bin_num, 2
        )
        r_ref1_results_1_4[i, :] = r_ref1_result_1_4
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "prev_results_1_4__{}_{}_{}.png".format(index, i, j),
                        ),
                        r_ref1_results_1_4[i, :, :, j] * 255,
                    )

        ref1_result_img_1_8 = ref1_result[i, :]
        r_ref1_result_1_8 = image_detach_with_id_color_probability_list(
            ref1_result_img_1_8, prev_id_color_list, bin_num, 3
        )
        r_ref1_results_1_8[i, :] = r_ref1_result_1_8
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "prev_results_1_8__{}_{}_{}.png".format(index, i, j),
                        ),
                        r_ref1_results_1_8[i, :, :, j] * 255,
                    )

        r_ref1_result = image_detach_with_id_color_probability_list(
            ref1_result[i, :], prev_id_color_list, bin_num, 0
        )
        r_ref1_results[i, :] = r_ref1_result
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "prev_results__{}_{}_{}.png".format(index, i, j),
                        ),
                        r_ref1_results[i, :, :, j] * 255,
                    )

        r_label = image_detach_with_id_color_list(
            label[i, :], prev_id_color_list, bin_num
        )
        r_labels[i, :] = r_label
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "target_labels_{}_{}_{}.png".format(index, i, j),
                        ),
                        r_labels[i, :, :, j] * 255,
                    )
    return (
        main_gray_img,
        ref1_gray_img,
        r_ref1_results,
        r_ref1_results_1_2,
        r_ref1_results_1_4,
        r_ref1_results_1_8,
        r_labels,
    )


def adjust_data4(
    main_gray_img: np.ndarray,
    ref1_gray_img: np.ndarray,
    ref1_result: np.ndarray,
    label: np.ndarray,
    bin_num: int,
    index: int = 0,
    save_to_dir_adjust: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    그레이스케일 이미지와 라벨 이미지의 값 범위를 조정합니다.

    Parameters
    ----------
    main_gray_img : np.ndarray
        현재 이미지 (그레이스케일)
    ref1_gray_img : np.ndarray
        참조 이미지 (그레이스케일)
    ref1_result : np.ndarray
        참조 결과 라벨 이미지 (컬러)
    label : np.ndarray
        라벨 이미지 (컬러)
    bin_num : int
        빈 개수

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        그레이스케일 이미지의 범위를 0~255에서, 0~1로 바꾸고, 라벨의 경우는 0~1 범위로 바꾼 후, threshold에 따라, 0과 1의 이진 이미지로 만듭니다.
    """
    batch_size: int = ref1_result.shape[0]
    main_gray_img = gray_image_to_ratio(main_gray_img)
    if save_to_dir_adjust is not None:
        for i in range(0, batch_size):
            cv2.imwrite(
                os.path.join(save_to_dir_adjust, "main_img_{}_{}.png".format(index, i)),
                main_gray_img[i, :] * 255,
            )

    ref1_gray_img = gray_image_to_ratio(ref1_gray_img)
    if save_to_dir_adjust is not None:
        for i in range(0, batch_size):
            cv2.imwrite(
                os.path.join(save_to_dir_adjust, "ref1_img_{}_{}.png".format(index, i)),
                ref1_gray_img[i, :] * 255,
            )

    batch_size: int = ref1_result.shape[0]
    r_ref1_results = np.zeros(ref1_result.shape[0:3] + (bin_num,))
    r_labels = np.zeros(label.shape[:3] + (bin_num,))
    r_ref1_size_1_2 = tuple(int(el / 2) for el in ref1_result.shape[1:3])
    r_ref1_results_1_2 = np.zeros(
        (ref1_result.shape[0],) + r_ref1_size_1_2 + (bin_num,)
    )
    r_ref1_size_1_4 = tuple(int(el / 4) for el in ref1_result.shape[1:3])
    r_ref1_results_1_4 = np.zeros(
        (ref1_result.shape[0],) + r_ref1_size_1_4 + (bin_num,)
    )
    for i in range(0, batch_size):
        ref1_label_img = ref1_result[i, :]
        ref1_label_img = ref1_label_img.astype(np.uint8)
        prev_color_list: List[Tuple[int, int, int]] = get_rgb_color_cv2(
            ref1_label_img, False
        )
        # print("prev_color_list: {}".format(prev_color_list))
        prev_id_color_list: List[Tuple[int, Tuple[int, int, int]]] = color_map_generate(
            prev_color_list
        )
        # print("prev_id_color_list: {}".format(prev_id_color_list))

        ref1_result_img_1_2 = ref1_result[i, :]
        r_ref1_result_1_2 = image_detach_with_id_color_probability_list(
            ref1_result_img_1_2, prev_id_color_list, bin_num, 1
        )
        r_ref1_results_1_2[i, :] = r_ref1_result_1_2
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "prev_results_1_2__{}_{}_{}.png".format(index, i, j),
                        ),
                        r_ref1_results_1_2[i, :, :, j] * 255,
                    )

        ref1_result_img_1_4 = ref1_result[i, :]
        r_ref1_result_1_4 = image_detach_with_id_color_probability_list(
            ref1_result_img_1_4, prev_id_color_list, bin_num, 2
        )
        r_ref1_results_1_4[i, :] = r_ref1_result_1_4
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "prev_results_1_4__{}_{}_{}.png".format(index, i, j),
                        ),
                        r_ref1_results_1_4[i, :, :, j] * 255,
                    )

        r_ref1_result = image_detach_with_id_color_probability_list(
            ref1_result[i, :], prev_id_color_list, bin_num, 0
        )
        r_ref1_results[i, :] = r_ref1_result
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "prev_results__{}_{}_{}.png".format(index, i, j),
                        ),
                        r_ref1_results[i, :, :, j] * 255,
                    )

        r_label = image_detach_with_id_color_list(
            label[i, :], prev_id_color_list, bin_num
        )
        r_labels[i, :] = r_label
        if save_to_dir_adjust is not None:
            for i in range(0, batch_size):
                for j in range(0, bin_num):
                    cv2.imwrite(
                        os.path.join(
                            save_to_dir_adjust,
                            "target_labels_{}_{}_{}.png".format(index, i, j),
                        ),
                        r_labels[i, :, :, j] * 255,
                    )
    return (
        main_gray_img,
        ref1_gray_img,
        r_ref1_results,
        r_ref1_results_1_2,
        r_ref1_results_1_4,
        r_labels,
    )


def image_label_set_generator(
    batch_size: int,
    directory: str,
    image_folder: str,
    main_image_folder: str,
    ref_1st_image_folder: str,
    ref_1st_result_folder: str,
    label_folder: str,
    image_data_generator: ImageDataGenerator,
    ref_1st_result_data_generator: ImageDataGenerator,
    label_data_generator: ImageDataGenerator,
    bin_num: int,
    image_color_mode: str = "grayscale",
    label_color_mode: str = "grayscale",
    image_save_prefix: str = "image",
    label_save_prefix: str = "label",
    save_to_dir=None,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 1,
):
    """
    이미지와 라벨을 동시에 생성합니다.

    Use the same seed for image_data_generator and label_data_generator to ensure the transformation for image and
    label is the same.
    If you want to visualize the results of generator, set save_to_dir = "your path"

    Parameters
    ----------
    batch_size : int
        데이터의 배치 사이즈
    directory : str
        타겟 폴더
    image_folder : str
        이미지 폴더
    ref_1st_image_folder : str
        첫 번째 참조 이미지 폴더
    ref_1st_result_folder : str
        첫 번째 참조 라벨 폴더
    label_folder : str
        라벨 폴더
    image_data_generator : ImageDataGenerator
        이미지 및 첫 번째 참조 이미지 ImageDataGenerator
    ref_1st_result_data_generator : ImageDataGenerator
        첫 번째 참조 라벨 ImageDataGenerator
    label_data_generator : ImageDataGenerator
        라벨 ImageDataGenerator
    image_color_mode : str, optional
        이미지 컬러 모드, by default "grayscale"
    label_color_mode : str, optional
        [description], by default "grayscale"
    image_save_prefix : str, optional
        [description], by default "image"
    label_save_prefix : str, optional
        [description], by default "label"
    save_to_dir : [type], optional
        [description], by default None
    target_size : Tuple[int, int], optional
        [description], by default (256, 256)
    seed : int, optional
        [description], by default 1

    Yields
    -------
    [type]
        이미지, 참조 이미지, 참조 라벨, 라벨 Generator
    """
    image_generator = image_data_generator.flow_from_directory(
        os.path.join(directory, image_folder),
        classes=[main_image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="main_{}".format(image_save_prefix),
        seed=seed,
    )
    ref_1st_image_generator = image_data_generator.flow_from_directory(
        os.path.join(directory, image_folder),
        classes=[ref_1st_image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="ref_1st_{}".format(image_save_prefix),
        seed=seed,
    )
    ref_1st_result_generator = ref_1st_result_data_generator.flow_from_directory(
        directory,
        classes=[ref_1st_result_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="ref_1st_result_{}".format(label_save_prefix),
        seed=seed,
    )
    label_generator = label_data_generator.flow_from_directory(
        directory,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed,
    )
    train_generators = zip(
        image_generator,
        ref_1st_image_generator,
        ref_1st_result_generator,
        label_generator,
    )
    for (img, ref_1st_image, ref_1st_result, label) in train_generators:
        img, ref_1st_image, ref_1st_result, label = adjust_data(
            img, ref_1st_image, ref_1st_result, label, bin_num
        )
        yield [img, ref_1st_image, ref_1st_result], label


def image_label_set_generator2(
    batch_size: int,
    directory: str,
    image_folder: str,
    main_image_folder: str,
    ref_1st_image_folder: str,
    ref_1st_result_folder: str,
    label_folder: str,
    image_data_generator: ImageDataGenerator,
    ref_1st_result_data_generator: ImageDataGenerator,
    label_data_generator: ImageDataGenerator,
    bin_num: int,
    image_color_mode: str = "grayscale",
    label_color_mode: str = "grayscale",
    image_save_prefix: str = "image",
    label_save_prefix: str = "label",
    save_to_dir=None,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 1,
):
    """
    이미지와 라벨을 동시에 생성합니다.

    Use the same seed for image_data_generator and label_data_generator to ensure the transformation for image and
    label is the same.
    If you want to visualize the results of generator, set save_to_dir = "your path"

    Parameters
    ----------
    batch_size : int
        데이터의 배치 사이즈
    directory : str
        타겟 폴더
    image_folder : str
        이미지 폴더
    ref_1st_image_folder : str
        첫 번째 참조 이미지 폴더
    ref_1st_result_folder : str
        첫 번째 참조 라벨 폴더
    label_folder : str
        라벨 폴더
    image_data_generator : ImageDataGenerator
        이미지 및 첫 번째 참조 이미지 ImageDataGenerator
    ref_1st_result_data_generator : ImageDataGenerator
        첫 번째 참조 라벨 ImageDataGenerator
    label_data_generator : ImageDataGenerator
        라벨 ImageDataGenerator
    image_color_mode : str, optional
        이미지 컬러 모드, by default "grayscale"
    label_color_mode : str, optional
        [description], by default "grayscale"
    image_save_prefix : str, optional
        [description], by default "image"
    label_save_prefix : str, optional
        [description], by default "label"
    save_to_dir : [type], optional
        [description], by default None
    target_size : Tuple[int, int], optional
        [description], by default (256, 256)
    seed : int, optional
        [description], by default 1

    Yields
    -------
    [type]
        이미지, 참조 이미지, 참조 라벨, 라벨 Generator
    """
    image_generator = image_data_generator.flow_from_directory(
        os.path.join(directory, image_folder),
        classes=[main_image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="main_{}".format(image_save_prefix),
        seed=seed,
    )
    ref_1st_image_generator = image_data_generator.flow_from_directory(
        os.path.join(directory, image_folder),
        classes=[ref_1st_image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="ref_1st_{}".format(image_save_prefix),
        seed=seed,
    )
    ref_1st_result_generator = ref_1st_result_data_generator.flow_from_directory(
        directory,
        classes=[ref_1st_result_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="ref_1st_result_{}".format(label_save_prefix),
        seed=seed,
    )
    label_generator = label_data_generator.flow_from_directory(
        directory,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed,
    )
    train_generators = zip(
        image_generator,
        ref_1st_image_generator,
        ref_1st_result_generator,
        label_generator,
    )
    for (img, ref_1st_image, ref_1st_result, label) in train_generators:
        (
            img,
            ref_1st_image,
            ref_1st_result,
            ref_1st_result_1_2,
            ref_1st_result_1_4,
            ref_1st_result_1_8,
            label,
        ) = adjust_data2(img, ref_1st_image, ref_1st_result, label, bin_num,)
        yield [
            img,
            ref_1st_image,
            ref_1st_result_1_8,
            ref_1st_result_1_4,
            ref_1st_result_1_2,
            ref_1st_result,
        ], label


def image_label_set_generator3(
    batch_size: int,
    directory: str,
    image_folder: str,
    main_image_folder: str,
    ref_1st_image_folder: str,
    ref_1st_result_folder: str,
    label_folder: str,
    image_data_generator: ImageDataGenerator,
    ref_1st_result_data_generator: ImageDataGenerator,
    label_data_generator: ImageDataGenerator,
    bin_num: int,
    image_color_mode: str = "grayscale",
    label_color_mode: str = "grayscale",
    image_save_prefix: str = "image",
    label_save_prefix: str = "label",
    save_to_dir=None,
    save_to_dir_adjust=None,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 1,
):
    """
    이미지와 라벨을 동시에 생성합니다.

    Use the same seed for image_data_generator and label_data_generator to ensure the transformation for image and
    label is the same.
    If you want to visualize the results of generator, set save_to_dir = "your path"

    Parameters
    ----------
    batch_size : int
        데이터의 배치 사이즈
    directory : str
        타겟 폴더
    image_folder : str
        이미지 폴더
    ref_1st_image_folder : str
        첫 번째 참조 이미지 폴더
    ref_1st_result_folder : str
        첫 번째 참조 라벨 폴더
    label_folder : str
        라벨 폴더
    image_data_generator : ImageDataGenerator
        이미지 및 첫 번째 참조 이미지 ImageDataGenerator
    ref_1st_result_data_generator : ImageDataGenerator
        첫 번째 참조 라벨 ImageDataGenerator
    label_data_generator : ImageDataGenerator
        라벨 ImageDataGenerator
    image_color_mode : str, optional
        이미지 컬러 모드, by default "grayscale"
    label_color_mode : str, optional
        [description], by default "grayscale"
    image_save_prefix : str, optional
        [description], by default "image"
    label_save_prefix : str, optional
        [description], by default "label"
    save_to_dir : [type], optional
        [description], by default None
    target_size : Tuple[int, int], optional
        [description], by default (256, 256)
    seed : int, optional
        [description], by default 1

    Yields
    -------
    [type]
        이미지, 참조 이미지, 참조 라벨, 라벨 Generator
    """
    image_generator = image_data_generator.flow_from_directory(
        os.path.join(directory, image_folder),
        classes=[main_image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="main_{}".format(image_save_prefix),
        seed=seed,
    )
    ref_1st_image_generator = image_data_generator.flow_from_directory(
        os.path.join(directory, image_folder),
        classes=[ref_1st_image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="ref_1st_{}".format(image_save_prefix),
        seed=seed,
    )
    ref_1st_result_generator = ref_1st_result_data_generator.flow_from_directory(
        directory,
        classes=[ref_1st_result_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="ref_1st_result_{}".format(label_save_prefix),
        seed=seed,
    )
    label_generator = label_data_generator.flow_from_directory(
        directory,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed,
    )
    train_generators = zip(
        image_generator,
        ref_1st_image_generator,
        ref_1st_result_generator,
        label_generator,
    )
    for index, (img, ref_1st_image, ref_1st_result, label) in enumerate(
        train_generators
    ):
        (
            img,
            ref_1st_image,
            ref_1st_result,
            ref_1st_result_1_2,
            ref_1st_result_1_4,
            ref_1st_result_1_8,
            label,
        ) = adjust_data3(
            img,
            ref_1st_image,
            ref_1st_result,
            label,
            bin_num,
            index,
            save_to_dir_adjust,
        )
        yield [
            img,
            ref_1st_image,
            ref_1st_result_1_8,
            ref_1st_result_1_4,
            ref_1st_result_1_2,
            ref_1st_result,
        ], label


def image_label_set_generator4(
    batch_size: int,
    directory: str,
    image_folder: str,
    main_image_folder: str,
    ref_1st_image_folder: str,
    ref_1st_result_folder: str,
    label_folder: str,
    image_data_generator: ImageDataGenerator,
    ref_1st_result_data_generator: ImageDataGenerator,
    label_data_generator: ImageDataGenerator,
    bin_num: int,
    image_color_mode: str = "grayscale",
    label_color_mode: str = "grayscale",
    image_save_prefix: str = "image",
    label_save_prefix: str = "label",
    save_to_dir=None,
    save_to_dir_adjust=None,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 1,
):
    """
    이미지와 라벨을 동시에 생성합니다.

    Use the same seed for image_data_generator and label_data_generator to ensure the transformation for image and
    label is the same.
    If you want to visualize the results of generator, set save_to_dir = "your path"

    Parameters
    ----------
    batch_size : int
        데이터의 배치 사이즈
    directory : str
        타겟 폴더
    image_folder : str
        이미지 폴더
    ref_1st_image_folder : str
        첫 번째 참조 이미지 폴더
    ref_1st_result_folder : str
        첫 번째 참조 라벨 폴더
    label_folder : str
        라벨 폴더
    image_data_generator : ImageDataGenerator
        이미지 및 첫 번째 참조 이미지 ImageDataGenerator
    ref_1st_result_data_generator : ImageDataGenerator
        첫 번째 참조 라벨 ImageDataGenerator
    label_data_generator : ImageDataGenerator
        라벨 ImageDataGenerator
    image_color_mode : str, optional
        이미지 컬러 모드, by default "grayscale"
    label_color_mode : str, optional
        [description], by default "grayscale"
    image_save_prefix : str, optional
        [description], by default "image"
    label_save_prefix : str, optional
        [description], by default "label"
    save_to_dir : [type], optional
        [description], by default None
    target_size : Tuple[int, int], optional
        [description], by default (256, 256)
    seed : int, optional
        [description], by default 1

    Yields
    -------
    [type]
        이미지, 참조 이미지, 참조 라벨, 라벨 Generator
    """
    image_generator = image_data_generator.flow_from_directory(
        os.path.join(directory, image_folder),
        classes=[main_image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="main_{}".format(image_save_prefix),
        seed=seed,
    )
    ref_1st_image_generator = image_data_generator.flow_from_directory(
        os.path.join(directory, image_folder),
        classes=[ref_1st_image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="ref_1st_{}".format(image_save_prefix),
        seed=seed,
    )
    ref_1st_result_generator = ref_1st_result_data_generator.flow_from_directory(
        directory,
        classes=[ref_1st_result_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix="ref_1st_result_{}".format(label_save_prefix),
        seed=seed,
    )
    label_generator = label_data_generator.flow_from_directory(
        directory,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=label_save_prefix,
        seed=seed,
    )
    train_generators = zip(
        image_generator,
        ref_1st_image_generator,
        ref_1st_result_generator,
        label_generator,
    )
    for index, (img, ref_1st_image, ref_1st_result, label) in enumerate(
        train_generators
    ):
        (
            img,
            ref_1st_image,
            ref_1st_result,
            ref_1st_result_1_2,
            ref_1st_result_1_4,
            label,
        ) = adjust_data4(
            img,
            ref_1st_image,
            ref_1st_result,
            label,
            bin_num,
            index,
            save_to_dir_adjust,
        )
        yield [
            img,
            ref_1st_image,
            ref_1st_result_1_4,
            ref_1st_result_1_2,
            ref_1st_result,
        ], label


def img_read_preprocessing(
    path_file: str,
    optional_pre_processing_function: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
    target_size=(256, 256),
    as_gray=True,
):
    img = io.imread(path_file, as_gray=as_gray)

    img = np.array(img, dtype=np.uint8)
    img = (
        optional_pre_processing_function(img)
        if optional_pre_processing_function is not None
        else img
    )
    img = gray_image_to_ratio(img)

    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)
    return img


def predict_generator(
    main_img_folder: str,
    ref_1st_img_folder: str,
    ref_1st_result_folder: str,
    ref_1st_result_detach_function: Callable[[np.ndarray, int], np.ndarray],
    bin_num: int,
    files: List[str],
    optional_pre_processing_function: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
    target_size=(256, 256),
    gray_image=True,
    gray_label=False,
):
    """
    [summary]

    Parameters
    ----------
    main_img_folder : str
        [description]
    ref_1st_img_folder : str
        [description]
    ref_1st_result_folder : str
        [description]
    files : List[str]
        [description]
    optional_pre_processing_function : Optional[ Callable[[np.ndarray], np.ndarray] ], optional
        [description], by default None
    target_size : tuple, optional
        [description], by default (256, 256)
    as_gray : bool, optional
        [description], by default True

    Yields
    -------
    [type]
        [description]
    """
    for file in files:
        main_img = io.imread(os.path.join(main_img_folder, file), as_gray=gray_image)
        main_img = np.array(main_img, dtype=np.uint8)
        if optional_pre_processing_function:
            main_img = optional_pre_processing_function(main_img)
        main_img = gray_image_to_ratio(main_img)
        main_img = trans.resize(main_img, target_size)
        main_img = np.reshape(main_img, main_img.shape + (1,))
        main_img = np.reshape(main_img, (1,) + main_img.shape)

        ref_1st_img = io.imread(
            os.path.join(ref_1st_img_folder, file), as_gray=gray_image
        )
        ref_1st_img = np.array(ref_1st_img, dtype=np.uint8)
        if optional_pre_processing_function:
            ref_1st_img = optional_pre_processing_function(ref_1st_img)
        ref_1st_img = gray_image_to_ratio(ref_1st_img)
        ref_1st_img = trans.resize(ref_1st_img, target_size)
        ref_1st_img = np.reshape(ref_1st_img, ref_1st_img.shape + (1,))
        ref_1st_img = np.reshape(ref_1st_img, (1,) + ref_1st_img.shape)

        ref_result_img = cv2.imread(os.path.join(ref_1st_result_folder, file))
        ref_result_img = cv2.resize(
            ref_result_img, target_size, interpolation=cv2.INTER_NEAREST
        )
        ref_result_img = ref_result_img.astype(np.uint8)
        ref_result_img = ref_1st_result_detach_function(ref_result_img, bin_num)
        ref_result_img = np.reshape(ref_result_img, (1,) + ref_result_img.shape)

        yield [main_img, ref_1st_img, ref_result_img]


def predict_generator2(
    main_img_folder: str,
    ref_1st_img_folder: str,
    ref_1st_result_folder: str,
    ref_1st_result_detach_function: Callable[[np.ndarray, int], np.ndarray],
    ref_1st_result_detach_function2: Callable[[np.ndarray, int], np.ndarray],
    bin_num: int,
    files: List[str],
    optional_pre_processing_function: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
    target_size=(256, 256),
    gray_image=True,
    gray_label=False,
):
    """
    [summary]

    Parameters
    ----------
    main_img_folder : str
        [description]
    ref_1st_img_folder : str
        [description]
    ref_1st_result_folder : str
        [description]
    files : List[str]
        [description]
    optional_pre_processing_function : Optional[ Callable[[np.ndarray], np.ndarray] ], optional
        [description], by default None
    target_size : tuple, optional
        [description], by default (256, 256)
    as_gray : bool, optional
        [description], by default True

    Yields
    -------
    [type]
        [description]
    """
    r_ref1_size_1_2 = tuple(int(el / 2) for el in target_size)
    r_ref1_size_1_4 = tuple(int(el / 4) for el in target_size)
    r_ref1_size_1_8 = tuple(int(el / 8) for el in target_size)

    for file in files:
        main_img = io.imread(os.path.join(main_img_folder, file), as_gray=gray_image)
        main_img = np.array(main_img, dtype=np.uint8)
        if optional_pre_processing_function:
            main_img = optional_pre_processing_function(main_img)
        main_img = gray_image_to_ratio(main_img)
        main_img = trans.resize(main_img, target_size)
        main_img = np.reshape(main_img, main_img.shape + (1,))
        main_img = np.reshape(main_img, (1,) + main_img.shape)

        ref_1st_img = io.imread(
            os.path.join(ref_1st_img_folder, file), as_gray=gray_image
        )
        ref_1st_img = np.array(ref_1st_img, dtype=np.uint8)
        if optional_pre_processing_function:
            ref_1st_img = optional_pre_processing_function(ref_1st_img)
        ref_1st_img = gray_image_to_ratio(ref_1st_img)
        ref_1st_img = trans.resize(ref_1st_img, target_size)
        ref_1st_img = np.reshape(ref_1st_img, ref_1st_img.shape + (1,))
        ref_1st_img = np.reshape(ref_1st_img, (1,) + ref_1st_img.shape)

        ref_result_img = cv2.imread(os.path.join(ref_1st_result_folder, file))
        ref_result_img = cv2.resize(
            ref_result_img, target_size, interpolation=cv2.INTER_NEAREST
        )
        ref_result_img = ref_result_img.astype(np.uint8)
        ref_result_img = ref_1st_result_detach_function(ref_result_img, bin_num)
        ref_result_img = np.reshape(ref_result_img, (1,) + ref_result_img.shape)

        ref_ori_img = cv2.imread(os.path.join(ref_1st_result_folder, file))

        ref1_result_img_1_2 = ref_ori_img.copy()
        ref1_result_img_1_2 = cv2.resize(
            ref1_result_img_1_2, r_ref1_size_1_2, interpolation=cv2.INTER_NEAREST
        )
        ref1_result_img_1_2 = ref_1st_result_detach_function2(
            ref1_result_img_1_2, bin_num
        )
        ref1_result_img_1_2 = np.reshape(
            ref1_result_img_1_2, (1,) + ref1_result_img_1_2.shape
        )

        ref1_result_img_1_4 = ref_ori_img.copy()
        ref1_result_img_1_4 = cv2.resize(
            ref1_result_img_1_4, r_ref1_size_1_4, interpolation=cv2.INTER_NEAREST
        )
        ref1_result_img_1_4 = ref_1st_result_detach_function2(
            ref1_result_img_1_4, bin_num
        )
        ref1_result_img_1_4 = np.reshape(
            ref1_result_img_1_4, (1,) + ref1_result_img_1_4.shape
        )

        ref1_result_img_1_8 = ref_ori_img.copy()
        ref1_result_img_1_8 = cv2.resize(
            ref1_result_img_1_8, r_ref1_size_1_8, interpolation=cv2.INTER_NEAREST
        )
        ref1_result_img_1_8 = ref_1st_result_detach_function2(
            ref1_result_img_1_8, bin_num
        )
        ref1_result_img_1_8 = np.reshape(
            ref1_result_img_1_8, (1,) + ref1_result_img_1_8.shape
        )

        yield [
            main_img,
            ref_1st_img,
            ref1_result_img_1_8,
            ref1_result_img_1_4,
            ref1_result_img_1_2,
            ref_result_img,
        ]


def predict_generator3(
    main_img_folder: str,
    ref_1st_img_folder: str,
    ref_1st_result_folder: str,
    ref_1st_result_detach_function: Callable[[np.ndarray, int], np.ndarray],
    ref_1st_result_detach_function1: Callable[[np.ndarray, int], np.ndarray],
    ref_1st_result_detach_function2: Callable[[np.ndarray, int], np.ndarray],
    ref_1st_result_detach_function3: Callable[[np.ndarray, int], np.ndarray],
    bin_num: int,
    files: List[str],
    optional_pre_processing_function: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
    target_size=(256, 256),
    gray_image=True,
    gray_label=False,
):
    """
    [summary]

    Parameters
    ----------
    main_img_folder : str
        [description]
    ref_1st_img_folder : str
        [description]
    ref_1st_result_folder : str
        [description]
    files : List[str]
        [description]
    optional_pre_processing_function : Optional[ Callable[[np.ndarray], np.ndarray] ], optional
        [description], by default None
    target_size : tuple, optional
        [description], by default (256, 256)
    as_gray : bool, optional
        [description], by default True

    Yields
    -------
    [type]
        [description]
    """
    r_ref1_size_1_2 = tuple(int(el / 2) for el in target_size)
    r_ref1_size_1_4 = tuple(int(el / 4) for el in target_size)
    r_ref1_size_1_8 = tuple(int(el / 8) for el in target_size)

    for file in files:
        main_img = io.imread(os.path.join(main_img_folder, file), as_gray=gray_image)
        main_img = np.array(main_img, dtype=np.uint8)
        if optional_pre_processing_function:
            main_img = optional_pre_processing_function(main_img)
        main_img = gray_image_to_ratio(main_img)
        main_img = trans.resize(main_img, target_size)
        main_img = np.reshape(main_img, main_img.shape + (1,))
        main_img = np.reshape(main_img, (1,) + main_img.shape)

        ref_1st_img = io.imread(
            os.path.join(ref_1st_img_folder, file), as_gray=gray_image
        )
        ref_1st_img = np.array(ref_1st_img, dtype=np.uint8)
        if optional_pre_processing_function:
            ref_1st_img = optional_pre_processing_function(ref_1st_img)
        ref_1st_img = gray_image_to_ratio(ref_1st_img)
        ref_1st_img = trans.resize(ref_1st_img, target_size)
        ref_1st_img = np.reshape(ref_1st_img, ref_1st_img.shape + (1,))
        ref_1st_img = np.reshape(ref_1st_img, (1,) + ref_1st_img.shape)

        ref_result_img = cv2.imread(os.path.join(ref_1st_result_folder, file))
        ref_result_img = cv2.resize(
            ref_result_img, target_size, interpolation=cv2.INTER_NEAREST
        )
        ref_result_img = ref_result_img.astype(np.uint8)
        ref_result_img = ref_1st_result_detach_function(ref_result_img, bin_num)
        ref_result_img = np.reshape(ref_result_img, (1,) + ref_result_img.shape)

        ref_ori_img = cv2.imread(os.path.join(ref_1st_result_folder, file))

        ref1_result_img_1_2 = ref_ori_img.copy()
        ref1_result_img_1_2 = ref_1st_result_detach_function1(
            ref1_result_img_1_2, bin_num
        )
        ref1_result_img_1_2 = np.reshape(
            ref1_result_img_1_2, (1,) + ref1_result_img_1_2.shape
        )

        ref1_result_img_1_4 = ref_ori_img.copy()
        ref1_result_img_1_4 = ref_1st_result_detach_function2(
            ref1_result_img_1_4, bin_num
        )
        ref1_result_img_1_4 = np.reshape(
            ref1_result_img_1_4, (1,) + ref1_result_img_1_4.shape
        )

        ref1_result_img_1_8 = ref_ori_img.copy()
        ref1_result_img_1_8 = ref_1st_result_detach_function3(
            ref1_result_img_1_8, bin_num
        )
        ref1_result_img_1_8 = np.reshape(
            ref1_result_img_1_8, (1,) + ref1_result_img_1_8.shape
        )

        yield [
            main_img,
            ref_1st_img,
            ref1_result_img_1_8,
            ref1_result_img_1_4,
            ref1_result_img_1_2,
            ref_result_img,
        ]


def predict_generator4(
    main_img_folder: str,
    ref_1st_img_folder: str,
    ref_1st_result_folder: str,
    ref_1st_result_detach_function: Callable[[np.ndarray, int], np.ndarray],
    ref_1st_result_detach_function1: Callable[[np.ndarray, int], np.ndarray],
    ref_1st_result_detach_function2: Callable[[np.ndarray, int], np.ndarray],
    ref_1st_result_detach_function3: Callable[[np.ndarray, int], np.ndarray],
    bin_num: int,
    files: List[str],
    optional_pre_processing_function: Optional[
        Callable[[np.ndarray], np.ndarray]
    ] = None,
    target_size=(256, 256),
    gray_image=True,
    gray_label=False,
    input_save_to_dir: str = None,
):
    """
    [summary]

    Parameters
    ----------
    main_img_folder : str
        [description]
    ref_1st_img_folder : str
        [description]
    ref_1st_result_folder : str
        [description]
    files : List[str]
        [description]
    optional_pre_processing_function : Optional[ Callable[[np.ndarray], np.ndarray] ], optional
        [description], by default None
    target_size : tuple, optional
        [description], by default (256, 256)
    as_gray : bool, optional
        [description], by default True

    Yields
    -------
    [type]
        [description]
    """
    r_ref1_size_1_2 = tuple(int(el / 2) for el in target_size)
    r_ref1_size_1_4 = tuple(int(el / 4) for el in target_size)
    r_ref1_size_1_8 = tuple(int(el / 8) for el in target_size)

    for file in files:
        main_img = cv2.imread(os.path.join(main_img_folder, file), cv2.IMREAD_GRAYSCALE)
        if optional_pre_processing_function:
            main_img = optional_pre_processing_function(main_img)
        main_img = gray_image_to_ratio(main_img)
        if input_save_to_dir:
            cv2.imwrite(
                os.path.join(input_save_to_dir, "main_img_" + file), main_img * 255
            )
        main_img = np.reshape(main_img, main_img.shape + (1,))
        main_img = np.reshape(main_img, (1,) + main_img.shape)

        ref_1st_img = cv2.imread(
            os.path.join(ref_1st_img_folder, file), cv2.IMREAD_GRAYSCALE
        )
        if optional_pre_processing_function:
            ref_1st_img = optional_pre_processing_function(ref_1st_img)
        ref_1st_img = gray_image_to_ratio(ref_1st_img)
        if input_save_to_dir:
            cv2.imwrite(
                os.path.join(input_save_to_dir, "ref_1st_img_" + file),
                ref_1st_img * 255,
            )
        ref_1st_img = np.reshape(ref_1st_img, ref_1st_img.shape + (1,))
        ref_1st_img = np.reshape(ref_1st_img, (1,) + ref_1st_img.shape)

        ref_result_img = cv2.imread(os.path.join(ref_1st_result_folder, file))
        ref_result_img = cv2.resize(
            ref_result_img, target_size, interpolation=cv2.INTER_NEAREST
        )
        ref_result_img = ref_result_img.astype(np.uint8)
        if input_save_to_dir:
            cv2.imwrite(
                os.path.join(input_save_to_dir, "ref_result_img_" + file),
                ref_result_img,
            )
        ref_result_img = ref_1st_result_detach_function(ref_result_img, bin_num)
        for bin_n in range(bin_num):
            if input_save_to_dir:
                cv2.imwrite(
                    os.path.join(
                        input_save_to_dir, "ref_result_img_{}_".format(bin_n) + file,
                    ),
                    ref_result_img[:, :, bin_n] * 255,
                )
        ref_result_img = np.reshape(ref_result_img, (1,) + ref_result_img.shape)

        ref_ori_img = cv2.imread(os.path.join(ref_1st_result_folder, file))

        ref1_result_img_1_2 = ref_ori_img.copy()
        if input_save_to_dir:
            cv2.imwrite(
                os.path.join(input_save_to_dir, "ref1_result_img_1_2_" + file),
                ref1_result_img_1_2,
            )
        ref1_result_img_1_2 = ref_1st_result_detach_function1(
            ref1_result_img_1_2, bin_num
        )
        for bin_n in range(bin_num):
            if input_save_to_dir:
                cv2.imwrite(
                    os.path.join(
                        input_save_to_dir,
                        "ref1_result_img_1_2_{}".format(bin_n) + file,
                    ),
                    ref1_result_img_1_2[:, :, bin_n] * 255,
                )
        ref1_result_img_1_2 = np.reshape(
            ref1_result_img_1_2, (1,) + ref1_result_img_1_2.shape
        )

        ref1_result_img_1_4 = ref_ori_img.copy()
        if input_save_to_dir:
            cv2.imwrite(
                os.path.join(input_save_to_dir, "ref1_result_img_1_4_" + file),
                ref1_result_img_1_4,
            )
        ref1_result_img_1_4 = ref_1st_result_detach_function2(
            ref1_result_img_1_4, bin_num
        )
        for bin_n in range(bin_num):
            if input_save_to_dir:
                cv2.imwrite(
                    os.path.join(
                        input_save_to_dir,
                        "ref1_result_img_1_4_{}".format(bin_n) + file,
                    ),
                    ref1_result_img_1_4[:, :, bin_n] * 255,
                )
        ref1_result_img_1_4 = np.reshape(
            ref1_result_img_1_4, (1,) + ref1_result_img_1_4.shape
        )

        yield [
            main_img,
            ref_1st_img,
            ref1_result_img_1_4,
            ref1_result_img_1_2,
            ref_result_img,
        ]


def save_result_with_name(
    save_path: str,
    numpy_images: List[np.ndarray],
    name_list: List[str],
    image_postprocessing_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    for index, numpy_image in enumerate(numpy_images):
        o = (255 * numpy_image[:, :, 0]).astype(np.uint8)
        o = (
            o
            if image_postprocessing_function is None
            else image_postprocessing_function(o)
        )
        io.imsave(os.path.join(save_path, name_list[index]), o, check_contrast=False)
