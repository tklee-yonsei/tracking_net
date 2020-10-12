from typing import Tuple

from layers.ref_local_layer import RefLocal
from models.gpu_check import check_first_gpu
from tensorflow.keras.layers import Conv2D, Input, Layer, UpSampling2D, concatenate
from tensorflow.keras.models import Model

check_first_gpu()

# example
# from models.semantic_segmentation.unet_l4.config import UnetL4ModelHelper
# unet_model_helper = UnetL4ModelHelper()
# unet_model = unet_model_helper.get_model()
# from models.ref_local_tracking.ref_local_tracking_model_001.config import (
#     RefModel001ModelHelper,
# )
# ref_model_helper = RefModel001ModelHelper(unet_model)
# model = ref_model_helper.get_model()


def ref_local_tracking_model_001(
    pre_trained_unet_l4_model: Model,
    input_main_image_name: str,
    input_main_image_shape: Tuple[int, int, int],
    input_ref_image_name: str,
    input_ref_image_shape: Tuple[int, int, int],
    input_ref_label_name: str,
    input_ref_label_shape: Tuple[int, int, int],
    output_name: str,
    bin_num: int = 30,
    alpha=1.0,
):
    filters: int = 16

    # U-Net
    pre_trained_unet_l4_model.trainable = False
    for layer in pre_trained_unet_l4_model.layers:
        layer.trainable = False

    # U-Net Skips
    skip_names = [
        pre_trained_unet_l4_model.layers[12].name,
        pre_trained_unet_l4_model.layers[8].name,
        pre_trained_unet_l4_model.layers[5].name,
        pre_trained_unet_l4_model.layers[2].name,
        pre_trained_unet_l4_model.layers[27].name,
    ]

    unet_l4_skip_1_model = Model(
        inputs=pre_trained_unet_l4_model.input,
        outputs=pre_trained_unet_l4_model.get_layer(skip_names[0]).output,
    )
    unet_l4_skip_2_model = Model(
        inputs=pre_trained_unet_l4_model.input,
        outputs=pre_trained_unet_l4_model.get_layer(skip_names[1]).output,
    )
    unet_l4_skip_3_model = Model(
        inputs=pre_trained_unet_l4_model.input,
        outputs=pre_trained_unet_l4_model.get_layer(skip_names[2]).output,
    )
    unet_l4_skip_4_model = Model(
        inputs=pre_trained_unet_l4_model.input,
        outputs=pre_trained_unet_l4_model.get_layer(skip_names[3]).output,
    )
    unet_l4_skip_result_model = Model(
        inputs=pre_trained_unet_l4_model.input,
        outputs=pre_trained_unet_l4_model.get_layer(skip_names[4]).output,
    )

    # 입력
    main_image_input: Layer = Input(
        shape=input_main_image_shape, name=input_main_image_name
    )
    ref_image_input: Layer = Input(
        shape=input_ref_image_shape, name=input_ref_image_name
    )
    ref_label_input: Layer = Input(
        shape=input_ref_label_shape, name=input_ref_label_name
    )

    # First
    main_l4_1 = unet_l4_skip_1_model(main_image_input)
    main_l4_1 = Conv2D(
        64, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    )(main_l4_1)
    ref_l4_1 = unet_l4_skip_1_model(ref_image_input)
    ref_l4_1 = Conv2D(
        64, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    )(ref_l4_1)
    ref_local_l4_1: Layer = RefLocal(mode="dot", k_size=5, bin_size=bin_num)(
        [main_l4_1, ref_l4_1, ref_label_input]
    )
    l4_up_1 = UpSampling2D(interpolation="bilinear")(ref_local_l4_1)
    # l4_up_1 = Conv2D(
    #     512, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    # )(l4_up_1)

    # Second
    main_l4_2 = unet_l4_skip_2_model(main_image_input)
    main_l4_2 = Conv2D(
        64, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    )(main_l4_2)
    ref_l4_2 = unet_l4_skip_2_model(ref_image_input)
    ref_l4_2 = Conv2D(
        64, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    )(ref_l4_2)
    ref_local_l4_2: Layer = RefLocal(mode="dot", k_size=5, bin_size=bin_num)(
        [main_l4_2, ref_l4_2, l4_up_1]
    )
    l4_up_2 = UpSampling2D(interpolation="bilinear")(ref_local_l4_2)

    # Third
    main_l4_3 = unet_l4_skip_3_model(main_image_input)
    main_l4_3 = Conv2D(
        64, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    )(main_l4_3)
    ref_l4_3 = unet_l4_skip_3_model(ref_image_input)
    ref_l4_3 = Conv2D(
        64, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    )(ref_l4_3)
    ref_local_l4_3: Layer = RefLocal(mode="dot", k_size=5, bin_size=bin_num)(
        [main_l4_3, ref_l4_3, l4_up_2]
    )
    l4_up_3 = UpSampling2D(interpolation="bilinear")(ref_local_l4_3)

    # Fourth
    main_l4_4 = unet_l4_skip_4_model(main_image_input)
    ref_l4_4 = unet_l4_skip_4_model(ref_image_input)
    ref_local_l4_4: Layer = RefLocal(mode="dot", k_size=5, bin_size=bin_num)(
        [main_l4_4, ref_l4_4, l4_up_3]
    )

    # Merge
    main_l4_res = unet_l4_skip_result_model(main_image_input)
    # main_l4_res = Conv2D(
    #     113, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    # )(main_l4_res)
    ref_l4_res = unet_l4_skip_result_model(ref_image_input)
    # ref_l4_res = Conv2D(
    #     113, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    # )(ref_l4_res)
    merge_layer: Layer = concatenate([main_l4_res, ref_l4_res, ref_local_l4_4], axis=3)

    merge_layer = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(merge_layer)
    merge_layer = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(merge_layer)

    total_conv: Layer = Conv2D(
        bin_num,
        1,
        activation="softmax",
        name=output_name,
        padding="same",
        kernel_initializer="he_normal",
    )(merge_layer)

    return Model(
        inputs=[main_image_input, ref_image_input, ref_label_input],
        outputs=[total_conv],
    )
