from typing import Tuple

from keras import losses, optimizers
from keras.layers import Conv2D, Input, Layer, UpSampling2D, concatenate
from keras.models import Model

from models.gpu_check import check_first_gpu

check_first_gpu()


def model_006(
    pre_trained_unet_l4_model: Model,
    input_main_image_name: str,
    input_main_image_shape: Tuple[int, int, int],
    input_ref_image_name: str,
    input_ref_image_shape: Tuple[int, int, int],
    input_ref1_label_name: str,
    input_ref1_label_shape: Tuple[int, int, int],
    input_ref2_label_name: str,
    input_ref2_label_shape: Tuple[int, int, int],
    input_ref3_label_name: str,
    input_ref3_label_shape: Tuple[int, int, int],
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
        pre_trained_unet_l4_model.layers[17].name,
        pre_trained_unet_l4_model.layers[5].name,
        pre_trained_unet_l4_model.layers[2].name,
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

    # 입력
    main_image_input: Layer = Input(
        shape=input_main_image_shape, name=input_main_image_name
    )
    ref_image_input: Layer = Input(
        shape=input_ref_image_shape, name=input_ref_image_name
    )
    ref1_label_input: Layer = Input(
        shape=input_ref1_label_shape, name=input_ref1_label_name
    )
    ref2_label_input: Layer = Input(
        shape=input_ref2_label_shape, name=input_ref2_label_name
    )
    ref3_label_input: Layer = Input(
        shape=input_ref3_label_shape, name=input_ref3_label_name
    )

    # Second
    ref_l4_2 = unet_l4_skip_1_model(ref_image_input)
    main_l4_2 = unet_l4_skip_1_model(main_image_input)
    l4_merge_2: Layer = concatenate([ref_l4_2, main_l4_2, ref1_label_input], axis=3)
    l4_conv_2 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(l4_merge_2)
    l4_conv_2 = Conv2D(
        256, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(l4_conv_2)
    l4_conv_2 = Conv2D(
        bin_num,
        1,
        activation="softmax",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_2)
    l4_conv_up_2 = UpSampling2D(interpolation="bilinear")(l4_conv_2)

    # Third
    ref1_l4_3 = unet_l4_skip_2_model(ref_image_input)
    main_l4_3 = unet_l4_skip_2_model(main_image_input)
    l4_merge_3: Layer = concatenate(
        [l4_conv_up_2, ref1_l4_3, main_l4_3, ref2_label_input], axis=3
    )
    l4_conv_3 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(l4_merge_3)
    l4_conv_3 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(l4_conv_3)
    l4_conv_3 = Conv2D(
        bin_num,
        1,
        activation="softmax",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_3)
    l4_conv_up_3 = UpSampling2D(interpolation="bilinear")(l4_conv_3)

    # Fourth
    ref1_l4_4 = unet_l4_skip_3_model(ref_image_input)
    main_l4_4 = unet_l4_skip_3_model(main_image_input)
    l4_merge_4: Layer = concatenate(
        [l4_conv_up_3, ref1_l4_4, main_l4_4, ref3_label_input], axis=3
    )
    l4_conv_4 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(l4_merge_4)
    l4_conv_4 = Conv2D(
        64, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(l4_conv_4)

    # ref1 네트워크
    total_conv: Layer = Conv2D(
        bin_num,
        1,
        activation="softmax",
        name=output_name,
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_4)

    return Model(
        inputs=[
            main_image_input,
            ref_image_input,
            ref1_label_input,
            ref2_label_input,
            ref3_label_input,
        ],
        outputs=[total_conv],
    )
