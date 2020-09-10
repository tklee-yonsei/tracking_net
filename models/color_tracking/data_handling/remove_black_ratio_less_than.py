import os
from argparse import ArgumentParser
import shutil
from typing import List

import common_py
import cv2

from .uu import count_bgr_color_pixels

# python -m color_tracking.data_handling.remove_black_ratio_less_than --dataset_name=ivan_training --target_dataset_name=ivan_filtered_training --black_ratio_more_than=0.0
# python -m color_tracking.data_handling.remove_black_ratio_less_than --dataset_name=ivan_test --target_dataset_name=ivan_filtered_test --black_ratio_more_than=0.0
# python -m color_tracking.data_handling.remove_black_ratio_less_than --dataset_name=ivan_validation --target_dataset_name=ivan_filtered_validation --black_ratio_more_than=0.0

if __name__ == "__main__":
    # -- 1. 명령 파라미터 ----
    parser: ArgumentParser = ArgumentParser(description="Color tracking net 데이터 정리기")

    # Default 값
    default_base_dataset_folder: str = "data"
    default_base_criteria_folder: str = "prev_result"
    default_prev_image_folder: str = os.path.join("image", "prev")
    default_current_image_folder: str = os.path.join("image", "current")
    default_target_label_folder: str = "label"
    default_black_ratio_less_than: float = 0.9
    default_black_ratio_more_than: float = -0.01

    # 데이터 준비
    parser.add_argument(
        "--dataset_name", required=True, type=str, help="데이터 세트 이름. 예 : ivan_training",
    )
    parser.add_argument(
        "--target_dataset_name",
        required=True,
        type=str,
        help="저장될 데이터 세트 이름. 예 : ivan_filtered_training",
    )
    parser.add_argument(
        "--base_dataset_folder",
        default=default_base_dataset_folder,
        type=str,
        help="데이터 세트 모음 폴더. 예 : data",
    )
    parser.add_argument(
        "--criteria_folder",
        default=default_base_criteria_folder,
        type=str,
        help="데이터 세트 이름. 예 : prev_result",
    )
    parser.add_argument(
        "--black_ratio_less_than",
        default=default_black_ratio_less_than,
        type=float,
        help="데이터 세트 이름. 예 : 0.9",
    )
    parser.add_argument(
        "--black_ratio_more_than",
        default=default_black_ratio_more_than,
        type=float,
        help="데이터 세트 이름. 예 : -0.01",
    )

    # -- 2. Argument 파싱 ----
    args = parser.parse_args()
    # 데이터 세트 ----
    # 데이터 세트 베이스 폴더
    base_dataset_folder: str = args.base_dataset_folder
    # 데이터 세트 이름
    dataset_name: str = args.dataset_name
    # 타겟 데이터 세트 이름
    target_dataset_name: str = args.target_dataset_name
    # 기준 폴더
    criteria_folder: bool = args.criteria_folder
    # 기준 threshold less than
    black_ratio_less_than: float = args.black_ratio_less_than
    # 기준 threshold more than
    black_ratio_more_than: float = args.black_ratio_more_than
    # 원본 트레이닝 데이터 폴더
    prev_image_folder: str = default_prev_image_folder
    # 원본 검증 데이터 폴더
    current_image_folder: str = default_current_image_folder
    # 원본 테스트 데이터 폴더
    target_label_folder: str = default_target_label_folder

    # -- 3.
    # 기준 폴더 파일 목록
    criteria_files = sorted(
        common_py.files_in_folder(
            os.path.join(base_dataset_folder, dataset_name, criteria_folder)
        )
    )
    # 이미지 읽고, 비율 계산 후, 리스트 추가
    criteria_ratios: List[float] = []
    for criteria_file in criteria_files:
        img = cv2.imread(
            os.path.join(
                base_dataset_folder, dataset_name, criteria_folder, criteria_file
            )
        )
        criteria_file_black_color = count_bgr_color_pixels(img)
        criteria_file_black_ratio = criteria_file_black_color / (
            img.shape[1] * img.shape[0]
        )
        criteria_ratios.append(criteria_file_black_ratio)
    criteria_file_ratios = list(zip(criteria_files, criteria_ratios))
    criteria_file_ratios = list(
        filter(lambda el: el[1] < black_ratio_less_than, criteria_file_ratios)
    )
    criteria_file_ratios = list(
        filter(lambda el: el[1] > black_ratio_more_than, criteria_file_ratios)
    )

    # 타겟 폴더 생성
    original_base = os.path.join(base_dataset_folder, dataset_name)
    target_base = os.path.join(base_dataset_folder, target_dataset_name)
    l = [criteria_folder, prev_image_folder, current_image_folder, target_label_folder]
    for el in l:
        common_py.create_folder(os.path.join(target_base, el))
        for criteria_file_ratio in criteria_file_ratios:
            print("{} // {}".format(el, criteria_file_ratio))
            shutil.copy2(
                os.path.join(original_base, el, criteria_file_ratio[0]),
                os.path.join(target_base, el),
            )
        print("{} {} files generated.".format(el, len(criteria_file_ratios)))
