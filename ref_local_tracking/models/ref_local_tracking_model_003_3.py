from typing import Tuple

from layers.ref_local_layer import RefLocal
from layers.ref_local_layer2 import RefLocal2
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Input,
    Layer,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model


def ref_local_tracking_model_003_3(
    pre_trained_unet_l4_model: Model,
    input_main_image_name: str,
    input_main_image_shape: Tuple[int, int, int],
    input_ref_image_name: str,
    input_ref_image_shape: Tuple[int, int, int],
    input_ref_label_1_name: str,
    input_ref_label_1_shape: Tuple[int, int, int],
    input_ref_label_2_name: str,
    input_ref_label_2_shape: Tuple[int, int, int],
    input_ref_label_3_name: str,
    input_ref_label_3_shape: Tuple[int, int, int],
    input_ref_label_4_name: str,
    input_ref_label_4_shape: Tuple[int, int, int],
    output_name: str,
    bin_num: int = 30,
    alpha: float = 1.0,
    unet_trainable: bool = True,
):
    filters: int = 16

    # U-Net
    pre_trained_unet_l4_model.trainable = unet_trainable
    for layer in pre_trained_unet_l4_model.layers:
        layer.trainable = unet_trainable

    # U-Net Skips
    skip_names = [
        pre_trained_unet_l4_model.layers[11].name,
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
    ref_label_1_input: Layer = Input(
        shape=input_ref_label_1_shape, name=input_ref_label_1_name
    )
    ref_label_2_input: Layer = Input(
        shape=input_ref_label_2_shape, name=input_ref_label_2_name
    )
    ref_label_3_input: Layer = Input(
        shape=input_ref_label_3_shape, name=input_ref_label_3_name
    )
    ref_label_4_input: Layer = Input(
        shape=input_ref_label_4_shape, name=input_ref_label_4_name
    )

    # First
    main_l4_1 = unet_l4_skip_1_model(main_image_input)
    ref_l4_1 = unet_l4_skip_1_model(ref_image_input)
    ref_local_l4_1: Layer = RefLocal(mode="dot", k_size=5, bin_size=bin_num)(
        [main_l4_1, ref_l4_1, ref_label_1_input]
    )
    # Temp added layers
    l4_up_1 = UpSampling2D(interpolation="bilinear")(ref_local_l4_1)
    # l4_up_1 = Conv2DTranspose(30, 3, strides=2, padding="same", kernel_initializer="he_normal")(ref_local_l4_1)

    # Second
    main_l4_2 = unet_l4_skip_2_model(main_image_input)
    ref_l4_2 = unet_l4_skip_2_model(ref_image_input)
    ref_local_l4_2: Layer = RefLocal2(mode="dot", k_size=5, bin_size=bin_num)(
        [main_l4_2, l4_up_1, ref_l4_2, ref_label_2_input]
    )
    l4_up_2 = UpSampling2D(interpolation="bilinear")(ref_local_l4_2)
    # l4_up_2 = Conv2DTranspose(30, 3, strides=2, padding="same", kernel_initializer="he_normal")(l4_up_1)

    # Third
    main_l4_3 = unet_l4_skip_3_model(main_image_input)
    ref_l4_3 = unet_l4_skip_3_model(ref_image_input)
    ref_local_l4_3: Layer = RefLocal2(mode="dot", k_size=5, bin_size=bin_num)(
        [main_l4_3, l4_up_2, ref_l4_3, ref_label_3_input]
    )
    l4_up_3 = UpSampling2D(interpolation="bilinear")(ref_local_l4_3)
    # l4_up_3 = Conv2DTranspose(30, 3, strides=2, padding="same", kernel_initializer="he_normal")(l4_up_2)

    # Fourth
    main_l4_4 = unet_l4_skip_4_model(main_image_input)
    ref_l4_4 = unet_l4_skip_4_model(ref_image_input)
    ref_local_l4_4: Layer = RefLocal2(mode="dot", k_size=5, bin_size=bin_num)(
        [main_l4_4, l4_up_3, ref_l4_4, ref_label_4_input]
    )

    total_conv: Layer = Conv2D(
        bin_num,
        1,
        activation="softmax",
        name=output_name,
        padding="same",
        kernel_initializer="he_normal",
    )(ref_local_l4_4)

    return Model(
        inputs=[
            main_image_input,
            ref_image_input,
            ref_label_1_input,
            ref_label_2_input,
            ref_label_3_input,
            ref_label_4_input,
        ],
        outputs=[total_conv],
    )
