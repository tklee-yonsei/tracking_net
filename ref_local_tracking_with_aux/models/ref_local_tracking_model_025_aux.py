from typing import Tuple

from layers.aggregation_layer import AggregationLayer
from layers.ref_local_layer8 import RefLocal8
from tensorflow.keras.layers import Conv2D, Input, Layer, UpSampling2D, concatenate
from tensorflow.keras.models import Model


def get_unet_layer(unet_model: Model):
    skip_names = [
        unet_model.layers[11].name,
        unet_model.layers[8].name,
        unet_model.layers[5].name,
        unet_model.layers[2].name,
        unet_model.layers[17].name,
        unet_model.layers[22].name,
        unet_model.layers[27].name,
    ]
    model = Model(
        inputs=unet_model.input,
        outputs=[
            unet_model.get_layer(skip_names[0]).output,
            unet_model.get_layer(skip_names[1]).output,
            unet_model.get_layer(skip_names[2]).output,
            unet_model.get_layer(skip_names[3]).output,
            unet_model.get_layer(skip_names[4]).output,
            unet_model.get_layer(skip_names[5]).output,
            unet_model.get_layer(skip_names[6]).output,
            unet_model.output,
        ],
    )
    return model


def ref_local_tracking_model_025_aux(
    unet_l4_model_main: Model,
    unet_l4_model_ref: Model,
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

    main_unet_model = get_unet_layer(unet_l4_model_main)
    ref_unet_model = get_unet_layer(unet_l4_model_ref)

    # Inputs
    main_image_input: Layer = Input(shape=input_main_image_shape)
    ref_image_input: Layer = Input(shape=input_ref_image_shape)
    ref_label_1_input: Layer = Input(shape=input_ref_label_1_shape)
    ref_label_2_input: Layer = Input(shape=input_ref_label_2_shape)
    ref_label_3_input: Layer = Input(shape=input_ref_label_3_shape)
    ref_label_4_input: Layer = Input(shape=input_ref_label_4_shape)

    main_unet = main_unet_model(main_image_input)
    ref_unet = ref_unet_model(ref_image_input)

    # First
    main1 = main_unet[0]
    ref1 = ref_unet[0]
    diff_local1: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=256)(
        [main1, ref1]
    )
    diff_agg1 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local1, ref_label_1_input])

    # Second
    main2 = main_unet[1]
    ref2 = ref_unet[1]
    diff_local2: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=128)(
        [main2, ref2]
    )
    diff_agg2 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local2, ref_label_2_input])

    # Third
    main3 = main_unet[2]
    ref3 = ref_unet[2]
    diff_local3: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=64)(
        [main3, ref3]
    )
    diff_agg3 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local3, ref_label_3_input])

    # Fourth
    main4 = main_unet[3]
    ref4 = ref_unet[3]
    diff_local4: Layer = RefLocal8(mode="dot", k_size=5, intermediate_dim=32)(
        [main4, ref4]
    )
    diff_agg4 = AggregationLayer(
        bin_size=bin_num, k_size=5, aggregate_mode="weighted_sum"
    )([diff_local4, ref_label_4_input])

    merge1 = concatenate([main1, diff_agg1])
    conv1 = Conv2D(
        filters=30,
        kernel_size=1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge1)
    up1: Layer = UpSampling2D()(conv1)

    main5 = main_unet[4]
    merge2 = concatenate([up1, main5, diff_agg2])
    conv2 = Conv2D(
        filters=30,
        kernel_size=1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge2)
    up2: Layer = UpSampling2D()(conv2)

    main6 = main_unet[5]
    merge3 = concatenate([up2, main6, diff_agg3])
    conv3 = Conv2D(
        filters=30,
        kernel_size=1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge3)
    up3: Layer = UpSampling2D()(conv3)

    main7 = main_unet[6]
    merge4 = concatenate([up3, main7, diff_agg4])
    conv4 = Conv2D(
        filters=30,
        kernel_size=1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge4)

    # Outputs
    output = Conv2D(
        filters=30,
        kernel_size=1,
        activation="softmax",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4)

    return Model(
        inputs=[
            main_image_input,
            ref_image_input,
            ref_label_1_input,
            ref_label_2_input,
            ref_label_3_input,
            ref_label_4_input,
        ],
        outputs=[output, main_unet[7], ref_unet[7]],
    )
