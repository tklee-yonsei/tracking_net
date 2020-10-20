import os
import sys

sys.path.append(os.getcwd())

import time

import common_py
import cv2
from image_keras.utils.image_transform import ratio_to_img
from models.semantic_segmentation.unet_l4.config import (
    UnetL4ModelHelper,
    single_generator,
    single_input_main_image_preprocessing,
)

if __name__ == "__main__":
    # 0. Prepare
    # ----------

    # predict_id: 사용한 모델, Predict 날짜
    # 0.1 ID ---------
    model_name: str = "unet_l4"
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
        base_dataset_folder, "test_original_20_edge10"
    )
    # input - image
    predict_image_folder: str = os.path.join(predict_dataset_folder, "image")

    # 1. Model
    # --------
    # model -> compile
    # a) model (from python code)
    unet_model_helper = UnetL4ModelHelper()
    unet_model = unet_model_helper.get_model()

    # b) compile
    model = unet_model_helper.compile_model(unet_model)

    # c) load weights
    weights_path: str = os.path.join(
        save_weights_folder, "unet010.hdf5",
    )
    model.load_weights(weights_path)

    # 2. Dataset
    # ----------
    predict_batch_size: int = 1

    # a) files
    common_files = sorted(common_py.files_in_folder(predict_image_folder))

    for index, common_file in enumerate(common_files):
        print(
            "Predict {0} ({1}/{2})".format(common_file, index + 1, len(common_files),)
        )
        this_files = [common_file]

        # a) image
        main_img = cv2.imread(
            os.path.join(predict_image_folder, common_file), cv2.IMREAD_GRAYSCALE,
        )
        main_img = single_input_main_image_preprocessing(main_img)
        # save main image pre processing result
        # main_img = single_input_main_image_preprocessing(
        #     main_img, (os.path.join(predict_result_folder), "_main_" + common_file),
        # )

        predict_generator = single_generator(main_img)

        results = model.predict_generator(predict_generator, 1, verbose=1)

        # 결과 출력
        result = results[0]  # (256, 256, 1)
        result = ratio_to_img(result)
        full_path: str = os.path.join(predict_result_folder, common_file)
        cv2.imwrite(full_path, result)

