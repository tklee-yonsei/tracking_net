import os
import sys

sys.path.append(os.getcwd())


import time
from typing import Dict, List, Tuple

import common_py
import cv2
import numpy as np
from models.ref_local_tracking.ref_local_tracking_model_003 import (
    RefModel003ModelHelper,
)
from models.ref_local_tracking.ref_local_tracking_model_003.config import (
    generate_color_map,
    single_generator,
    single_input_main_image_preprocessing,
    single_input_ref_image_preprocessing,
    single_input_ref_result_1_preprocessing,
    single_input_ref_result_2_preprocessing,
    single_input_ref_result_3_preprocessing,
    single_input_ref_result_4_preprocessing,
)

if __name__ == "__main__":
    # 0. Prepare
    # ----------

    # predict_id: 사용한 모델, Predict 날짜
    # 0.1 ID ---------
    model_name: str = "ref_local_tracking_model_003"
    run_id: str = time.strftime("%Y%m%d-%H%M%S")
    predict_id: str = "_predict__model_{}__run_{}".format(model_name, run_id)

    # 0.2 Folder ---------

    # a) model, weights, result
    base_dataset_folder: str = os.path.join("dataset")
    base_data_folder: str = os.path.join("data")
    base_save_folder: str = os.path.join("save")
    save_models_folder: str = os.path.join(base_save_folder, "models")
    save_weights_folder: str = os.path.join(base_save_folder, "weights")
    predict_result_folder: str = os.path.join(base_data_folder, predict_id)
    common_py.create_folder(predict_result_folder)

    # b) dataset folders
    predict_dataset_folder: str = os.path.join(
        base_dataset_folder, "ivan_filtered_test"
    )
    # input - main image
    predict_main_image_folder: str = os.path.join(
        predict_dataset_folder, "image", "current"
    )
    # input - ref image
    predict_ref_image_folder: str = os.path.join(
        predict_dataset_folder, "image", "prev"
    )
    # input - ref result label
    predict_ref_result_label_folder: str = os.path.join(
        predict_dataset_folder, "prev_result"
    )

    # 1. Model
    # --------
    # model -> compile
    # a) model (from python code)
    from models.semantic_segmentation.unet_l4.config_005 import UnetL4ModelHelper

    unet_model_helper = UnetL4ModelHelper()
    unet_model = unet_model_helper.get_model()
    model_helper = RefModel003ModelHelper(pre_trained_unet_l4_model=unet_model)
    model = model_helper.get_model()

    # b) compile
    model = model_helper.compile_model(model)

    # c) load weights
    weights_path: str = os.path.join(
        save_weights_folder,
        "training__model_ref_local_tracking_model_003__config_001__run_20201022-112411.epoch_105-val_loss_0.076-val_acc_0.980.hdf5",
    )
    model.load_weights(weights_path)

    # 2. Dataset
    # ----------
    predict_batch_size: int = 1

    # a) files
    common_files = sorted(common_py.files_in_folder(predict_main_image_folder))

    for index, common_file in enumerate(common_files):
        print(
            "Predict {0} ({1}/{2})".format(common_file, index + 1, len(common_files),)
        )
        this_files = [common_file]

        # 결과 파일에서 bin을 컬러 이미지로 복원하기 위한 빈 컬러 맵
        prev_result_img = cv2.imread(
            os.path.join(predict_ref_result_label_folder, common_file)
        )
        prev_id_color_list: List[Tuple[int, Tuple[int, int, int]]] = generate_color_map(
            prev_result_img, True, True
        )
        print(prev_id_color_list)

        # a) main image
        main_img = cv2.imread(
            os.path.join(predict_main_image_folder, common_file), cv2.IMREAD_GRAYSCALE,
        )
        main_img = single_input_main_image_preprocessing(main_img)
        # save main image pre processing result
        # main_img = single_input_main_image_preprocessing(
        #     main_img, (os.path.join(predict_result_folder), "_main_" + common_file),
        # )

        # b) ref image
        ref_img = cv2.imread(
            os.path.join(predict_ref_image_folder, common_file), cv2.IMREAD_GRAYSCALE,
        )
        ref_img = single_input_ref_image_preprocessing(ref_img)
        # save ref image pre processing result
        # ref_img = single_input_ref_image_preprocessing(
        #     ref_img, (os.path.join(predict_result_folder), "_ref_" + common_file),
        # )

        # c) ref result
        ref_result = cv2.imread(
            os.path.join(predict_ref_result_label_folder, common_file)
        )
        ref1_result = single_input_ref_result_1_preprocessing(ref_result)
        ref2_result = single_input_ref_result_2_preprocessing(ref_result)
        ref3_result = single_input_ref_result_3_preprocessing(ref_result)
        ref4_result = single_input_ref_result_4_preprocessing(ref_result)
        # save ref results pre processing result
        # ref1_result = single_input_ref1_result_preprocessing(
        #     ref_result,
        #     (os.path.join(predict_result_folder), "_ref1_result_" + common_file),
        # )
        # ref2_result = single_input_ref2_result_preprocessing(
        #     ref_result,
        #     (os.path.join(predict_result_folder), "_ref2_result_" + common_file),
        # )
        # ref3_result = single_input_ref3_result_preprocessing(
        #     ref_result,
        #     (os.path.join(predict_result_folder), "_ref3_result_" + common_file),
        # )

        predict_generator = single_generator(
            main_img, ref_img, ref1_result, ref2_result, ref3_result, ref4_result
        )

        results = model.predict_generator(
            predict_generator, 1, verbose=1, max_queue_size=1
        )

        # 결과 출력
        result = results[0]  # (256, 256, 30)

        # 1) np 파일로 결과 저장
        # np.save(
        #     os.path.join(predict_result_folder, common_file[: common_file.rfind(".")],),
        #     result,
        # )

        # 2) 각 빈에 대한 결과 출력.
        # t_name = list(map(lambda n: "{}".format(Path(n).stem), this_files))
        # for bin_id in range(result.shape[2]):
        #     img = result[:, :, bin_id : bin_id + 1]
        #     name_list = list(
        #         map(
        #             lambda n: "{}_{:02d}.png".format(Path(n).stem, bin_id + 1),
        #             this_files,
        #         )
        #     )
        #     img = img * 255
        #     cv2.imwrite(os.path.join(predict_result_folder, name_list[0]), img)

        def reduce_result(
            bin_probability_arr: np.ndarray,
            id_color_dict: Dict[int, Tuple[int, int, int]],
            default_value: Tuple[int, int, int],
        ) -> np.ndarray:
            argmax_arr = np.argmax(bin_probability_arr, axis=2)
            return np.asarray(
                [[id_color_dict.get(k, default_value) for k in k1] for k1 in argmax_arr]
            )

        # 3) 종합 결과 출력 (색상 복원 및 저장)
        color_img = reduce_result(result, dict(prev_id_color_list), (0, 0, 0))
        cv2.imwrite(os.path.join(predict_result_folder, this_files[0]), color_img)
