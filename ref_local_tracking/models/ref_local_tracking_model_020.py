import functools
from typing import Optional, Tuple

from layers.ref_local_layer5 import RefLocal5
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    Layer,
    Multiply,
    UpSampling2D,
    concatenate,
)
from tensorflow.keras.models import Model


def compose_left(*functions):
    return functools.reduce(lambda g, f: lambda x: f(g(x)), functions, lambda x: x)


def unet_base_conv_2d(
    filter_num: int,
    kernel_size: int = 3,
    activation="relu",
    padding="same",
    kernel_initializer="he_normal",
    name_optional: Optional[str] = None,
):
    return Conv2D(
        filters=filter_num,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        kernel_initializer=kernel_initializer,
        name=name_optional,
    )


def unet_base_up_sampling(
    filter_num: int,
    up_size: Tuple[int, int] = (2, 2),
    kernel_size: int = 3,
    activation="relu",
    padding="same",
    kernel_initializer="he_normal",
):
    up_sample_func = UpSampling2D(size=up_size)
    conv_func = Conv2D(
        filters=filter_num,
        kernel_size=kernel_size,
        activation=activation,
        padding=padding,
        kernel_initializer=kernel_initializer,
    )
    return compose_left(up_sample_func, conv_func)


def ref_local_tracking_model_020(
    pre_trained_unet_l4_model_main: Model,
    pre_trained_unet_l4_model_ref: Model,
    input_main_image_shape: Tuple[int, int, int] = (256, 256, 1),
    input_ref_image_shape: Tuple[int, int, int] = (256, 256, 1),
    input_ref_label_shape: Tuple[int, int, int] = (256, 256, 30),
    input_main_image_name: str = "main_image",
    input_ref_image_name: str = "ref_image",
    input_ref_label_name: str = "bin_label_1",
    output_name: str = "output",
    bin_num: int = 30,
    alpha: float = 1.0,
):
    filters: int = 16

    # U-Net Skips
    skip_names_main = [
        pre_trained_unet_l4_model_main.layers[11].name,
        pre_trained_unet_l4_model_main.layers[8].name,
        pre_trained_unet_l4_model_main.layers[5].name,
        pre_trained_unet_l4_model_main.layers[2].name,
    ]
    skip_names_ref = [
        pre_trained_unet_l4_model_ref.layers[11].name,
        pre_trained_unet_l4_model_ref.layers[8].name,
        pre_trained_unet_l4_model_ref.layers[5].name,
        pre_trained_unet_l4_model_ref.layers[2].name,
    ]

    unet_l4_skip_main_model_1 = Model(
        inputs=pre_trained_unet_l4_model_main.input,
        outputs=pre_trained_unet_l4_model_main.get_layer(skip_names_main[0]).output,
    )
    unet_l4_skip_ref_model_1 = Model(
        inputs=pre_trained_unet_l4_model_ref.input,
        outputs=pre_trained_unet_l4_model_ref.get_layer(skip_names_ref[0]).output,
    )

    unet_l4_skip_main_model_2 = Model(
        inputs=pre_trained_unet_l4_model_main.input,
        outputs=pre_trained_unet_l4_model_main.get_layer(skip_names_main[1]).output,
    )
    unet_l4_skip_ref_model_2 = Model(
        inputs=pre_trained_unet_l4_model_ref.input,
        outputs=pre_trained_unet_l4_model_ref.get_layer(skip_names_ref[1]).output,
    )

    unet_l4_skip_main_model_3 = Model(
        inputs=pre_trained_unet_l4_model_main.input,
        outputs=pre_trained_unet_l4_model_main.get_layer(skip_names_main[2]).output,
    )
    unet_l4_skip_ref_model_3 = Model(
        inputs=pre_trained_unet_l4_model_ref.input,
        outputs=pre_trained_unet_l4_model_ref.get_layer(skip_names_ref[2]).output,
    )

    unet_l4_skip_main_model_4 = Model(
        inputs=pre_trained_unet_l4_model_main.input,
        outputs=pre_trained_unet_l4_model_main.get_layer(skip_names_main[3]).output,
    )
    unet_l4_skip_ref_model_4 = Model(
        inputs=pre_trained_unet_l4_model_ref.input,
        outputs=pre_trained_unet_l4_model_ref.get_layer(skip_names_ref[3]).output,
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
    main1 = unet_l4_skip_main_model_1(main_image_input)
    ref1 = unet_l4_skip_ref_model_1(ref_image_input)
    diff_local1: Layer = RefLocal5(mode="dot", k_size=5, intermediate_dim=256)(
        [main1, ref1]
    )
    up1: Layer = unet_base_up_sampling(256)(diff_local1)

    # Second
    main2 = unet_l4_skip_main_model_2(main_image_input)
    ref2 = unet_l4_skip_ref_model_2(ref_image_input)
    diff_local2: Layer = RefLocal5(mode="dot", k_size=5, intermediate_dim=128)(
        [main2, ref2]
    )
    merge1 = concatenate([diff_local2, up1])
    conv1: Layer = unet_base_conv_2d(256)(merge1)
    conv1: Layer = unet_base_conv_2d(256)(conv1)
    up2: Layer = unet_base_up_sampling(128)(conv1)

    # Third
    main3 = unet_l4_skip_main_model_3(main_image_input)
    ref3 = unet_l4_skip_ref_model_3(ref_image_input)
    diff_local3: Layer = RefLocal5(mode="dot", k_size=5, intermediate_dim=64)(
        [main3, ref3]
    )
    merge2 = concatenate([diff_local3, up2])
    conv2: Layer = unet_base_conv_2d(128)(merge2)
    conv2: Layer = unet_base_conv_2d(128)(conv2)
    up3: Layer = unet_base_up_sampling(64)(conv2)

    # Fourth
    main4 = unet_l4_skip_main_model_4(main_image_input)
    ref4 = unet_l4_skip_ref_model_4(ref_image_input)
    diff_local4: Layer = RefLocal5(mode="dot", k_size=5, intermediate_dim=32)(
        [main4, ref4]
    )
    merge3 = concatenate([diff_local4, up3])
    conv3: Layer = unet_base_conv_2d(64)(merge3)
    conv3: Layer = unet_base_conv_2d(64)(conv3)

    # Outputs
    conv4: Layer = Conv2D(
        bin_num,
        1,
        activation="softmax",
        name=output_name,
        padding="same",
        kernel_initializer="he_normal",
    )(conv3)

    output: Layer = Multiply()([conv4, ref_label_input])

    return Model(
        inputs=[main_image_input, ref_image_input, ref_label_input], outputs=[output],
    )
