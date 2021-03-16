from typing import Tuple

from layers.aggregation_layer import AggregationLayer
from layers.aggregation_layer2 import AggregationLayer2
from layers.ref_local_layer7 import RefLocal7
from layers.ref_local_layer8 import RefLocal8
from layers.shrink_layer import ShrinkLayer
from tensorflow.keras.layers import Conv2D, Input, Layer, UpSampling2D, concatenate
from tensorflow.keras.models import Model


def get_unet_detached_layer(unet_model: Model):
    skip_names = [
        unet_model.layers[14].name,
        unet_model.layers[10].name,
        unet_model.layers[6].name,
        unet_model.layers[2].name,
    ]
    model = Model(
        inputs=unet_model.input,
        outputs=[
            unet_model.get_layer(skip_names[0]).output,
            unet_model.get_layer(skip_names[1]).output,
            unet_model.get_layer(skip_names[2]).output,
            unet_model.get_layer(skip_names[3]).output,
            unet_model.output,
        ],
    )
    return model


def aggregation_up(feature_map, filters: int):
    up_layer: Layer = UpSampling2D()(feature_map)
    up_conv_layer: Layer = Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        kernel_initializer="he_normal",
        activation="relu",
    )(up_layer)
    return up_conv_layer


def concat_conv(feature_map, filters: int):
    conv_layer: Layer = Conv2D(
        filters, 3, padding="same", kernel_initializer="he_normal"
    )(feature_map)
    conv_layer = Conv2D(256, 3, padding="same", kernel_initializer="he_normal")(
        conv_layer
    )
    return conv_layer


def ref_local_tracking_model_032(
    unet_l4_model_main: Model,
    unet_l4_model_ref: Model,
    input_main_image_shape: Tuple[int, int, int] = (256, 256, 1),
    input_ref_image_shape: Tuple[int, int, int] = (256, 256, 1),
    input_ref_label_shape: Tuple[int, int, int] = (256, 256, 30),
    bin_num: int = 30,
    alpha: float = 1.0,
):
    filters: int = 16

    main_unet_model = get_unet_detached_layer(unet_l4_model_main)
    ref_unet_model = get_unet_detached_layer(unet_l4_model_ref)

    # Inputs
    main_image_input: Layer = Input(shape=input_main_image_shape)
    ref_image_input: Layer = Input(shape=input_ref_image_shape)
    ref_label_input: Layer = Input(shape=input_ref_label_shape)

    ref_label_1_input = ShrinkLayer(bin_num=bin_num, resize_by_power_of_two=3)(
        ref_label_input
    )
    ref_label_2_input = ShrinkLayer(bin_num=bin_num, resize_by_power_of_two=2)(
        ref_label_input
    )
    ref_label_3_input = ShrinkLayer(bin_num=bin_num, resize_by_power_of_two=1)(
        ref_label_input
    )
    ref_label_4_input = ShrinkLayer(bin_num=bin_num, resize_by_power_of_two=0)(
        ref_label_input
    )

    main_unet = main_unet_model(main_image_input)
    ref_unet = ref_unet_model(ref_image_input)

    # First
    diff_local1: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=256)(
        [main_unet[0], ref_unet[0]]
    )
    diff_agg1 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local1, ref_label_1_input])
    up1: Layer = aggregation_up(diff_agg1, filters=bin_num)
    concat1: Layer = concatenate([main_unet[1], up1])
    conv1: Layer = concat_conv(concat1, filters=256)
    conv1 = Conv2D(bin_num, kernel_size=1, kernel_initializer="he_normal")(conv1)

    # Second
    diff_local2: Layer = RefLocal7(mode="dot", k_size=5, intermediate_dim=128)(
        [main_unet[1], ref_unet[1]]
    )
    diff_agg2 = AggregationLayer2(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local2, ref_label_2_input, conv1])
    up2: Layer = aggregation_up(diff_agg2, filters=bin_num)
    concat2: Layer = concatenate([main_unet[2], up2])
    conv2: Layer = concat_conv(concat2, filters=128)
    conv2 = Conv2D(bin_num, kernel_size=1, kernel_initializer="he_normal")(conv2)

    # Third
    diff_local3: Layer = RefLocal7(mode="dot", k_size=5, intermediate_dim=64)(
        [main_unet[2], ref_unet[2]]
    )
    diff_agg3 = AggregationLayer2(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local3, ref_label_3_input, conv2])
    up3: Layer = aggregation_up(diff_agg3, filters=bin_num)
    concat3: Layer = concatenate([main_unet[3], up3])
    conv3: Layer = concat_conv(concat3, filters=64)
    conv3 = Conv2D(bin_num, kernel_size=1, kernel_initializer="he_normal")(conv3)

    # Fourth
    diff_local4: Layer = RefLocal7(mode="dot", k_size=5, intermediate_dim=32)(
        [main_unet[3], ref_unet[3]]
    )
    diff_agg4 = AggregationLayer2(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local4, ref_label_4_input, conv3])

    conv1 = Conv2D(
        filters=60,
        kernel_size=1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(diff_agg4)
    conv1 = Conv2D(
        filters=60,
        kernel_size=1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)

    # Outputs
    conv2 = Conv2D(
        filters=bin_num,
        kernel_size=1,
        activation="softmax",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)

    return Model(
        inputs=[main_image_input, ref_image_input, ref_label_input],
        outputs=[conv2, main_unet[4], ref_unet[4]],
    )
