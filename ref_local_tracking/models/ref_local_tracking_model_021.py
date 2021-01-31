from typing import Tuple

from layers.aggregation_layer import AggregationLayer
from layers.ref_local_layer8 import RefLocal8
from tensorflow.keras.layers import Conv2D, Input, Layer, UpSampling2D, concatenate
from tensorflow.keras.models import Model


def ref_local_tracking_model_021(
    pre_trained_unet_l4_model_main: Model,
    pre_trained_unet_l4_model_ref: Model,
    input_main_image_shape: Tuple[int, int, int] = (256, 256, 1),
    input_ref_image_shape: Tuple[int, int, int] = (256, 256, 1),
    input_ref_label_1_shape: Tuple[int, int, int] = (32, 32, 30),
    input_ref_label_2_shape: Tuple[int, int, int] = (64, 64, 30),
    input_ref_label_3_shape: Tuple[int, int, int] = (128, 128, 30),
    input_ref_label_4_shape: Tuple[int, int, int] = (256, 256, 30),
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

    # Inputs
    main_image_input: Layer = Input(shape=input_main_image_shape)
    ref_image_input: Layer = Input(shape=input_ref_image_shape)
    ref_label_1_input: Layer = Input(shape=input_ref_label_1_shape)
    ref_label_2_input: Layer = Input(shape=input_ref_label_2_shape)
    ref_label_3_input: Layer = Input(shape=input_ref_label_3_shape)
    ref_label_4_input: Layer = Input(shape=input_ref_label_4_shape)

    # First
    main1 = unet_l4_skip_main_model_1(main_image_input)
    ref1 = unet_l4_skip_ref_model_1(ref_image_input)
    diff_local1: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=256)(
        [main1, ref1]
    )
    diff_agg1 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local1, ref_label_1_input])
    up1: Layer = UpSampling2D(size=8)(diff_agg1)

    # Second
    main2 = unet_l4_skip_main_model_2(main_image_input)
    ref2 = unet_l4_skip_ref_model_2(ref_image_input)
    diff_local2: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=128)(
        [main2, ref2]
    )
    diff_agg2 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local2, ref_label_2_input])
    up2: Layer = UpSampling2D(size=4)(diff_agg2)

    # Third
    main3 = unet_l4_skip_main_model_3(main_image_input)
    ref3 = unet_l4_skip_ref_model_3(ref_image_input)
    diff_local3: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=64)(
        [main3, ref3]
    )
    diff_agg3 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local3, ref_label_3_input])
    up3: Layer = UpSampling2D(size=2)(diff_agg3)

    # Fourth
    main4 = unet_l4_skip_main_model_4(main_image_input)
    ref4 = unet_l4_skip_ref_model_4(ref_image_input)
    diff_local4: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=32)(
        [main4, ref4]
    )
    diff_agg4 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local4, ref_label_4_input])

    merge1 = concatenate([up1, up2, up3, diff_agg4])
    conv1 = Conv2D(
        filters=60,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge1)
    conv1 = Conv2D(
        filters=60,
        kernel_size=3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)

    # Outputs
    conv2 = Conv2D(
        filters=30,
        kernel_size=3,
        activation="softmax",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)

    return Model(
        inputs=[
            main_image_input,
            ref_image_input,
            ref_label_1_input,
            ref_label_2_input,
            ref_label_3_input,
            ref_label_4_input,
        ],
        outputs=[conv2],
    )
