from typing import List, Tuple

import keras
from keras import losses, optimizers
from keras.layers import *
from keras.models import *

from seg_models.unet_l4.unet_l4 import unet_l4


def model(
    unet_pre_trained_weights: str,
    pre_trained_weights: str = None,
    bin_num: int = 30,
    input_size: Tuple[int, int, int] = (256, 256, 1),
) -> Model:
    """
    이전 및 현재 프레임의 이미지에서 각 인스턴스 객체를 bin 기반으로 추적하는 네트워크

    Parameters
    ----------
    unet_pre_trained_weights : str
        U-Net으로 사전 트레이닝된 네트워크 가중치의 파일 경로. (`pre_trained_weights` 값이 있으면 적용 안됨.)
    pre_trained_weights : str, optional
        사전 트레이닝된 네트워크 가중치, by default None
    bin_num : int, optional
        추적할 bin의 개수, by default 30
    input_size : Tuple[int, int, int], optional
        입력 이미지 크기, by default (256, 256, 1)

    Returns
    -------
    Model
        색상 추적 U-Net 모델

    Examples
    --------
    모델 플롯
    >>> from model.pcn_concat_unet_with_unet_4th_ref_half.pcn_concat_unet_with_unet_4th import pcn_concat_unet
    >>> model = pcn_concat_unet(unet_pre_trained_weights="models/unet009.hdf5")
    >>> from keras.utils import plot_model
    >>> plot_model(model, show_shapes=True, to_file='model.png', expand_nested=True, dpi=144)
    """
    # 파라미터
    unet_filters: int = 16
    filters: int = 4  # 메모리 용량의 한계로 인한 필터 수 감소

    # U-Net
    pre_trained_unet: Model = unet_l4(
        pre_trained_weight_path=unet_pre_trained_weights,
        input_size=input_size,
        filters=unet_filters,
    )
    pre_trained_unet.trainable = False
    for layer in pre_trained_unet.layers:
        layer.trainable = False

    # U-Net HyperColumn 정의
    hc_names = [
        pre_trained_unet.layers[12].name,
        pre_trained_unet.layers[17].name,
        pre_trained_unet.layers[22].name,
        pre_trained_unet.layers[27].name,
    ]
    l4_hc_model_1 = Model(
        inputs=pre_trained_unet.input,
        outputs=pre_trained_unet.get_layer(hc_names[0]).output,
    )
    l4_hc_model_2 = Model(
        inputs=pre_trained_unet.input,
        outputs=pre_trained_unet.get_layer(hc_names[1]).output,
    )
    l4_hc_model_3 = Model(
        inputs=pre_trained_unet.input,
        outputs=pre_trained_unet.get_layer(hc_names[2]).output,
    )
    l4_hc_model_4 = Model(
        inputs=pre_trained_unet.input,
        outputs=pre_trained_unet.get_layer(hc_names[3]).output,
    )

    # 입력
    main_input: Layer = Input(input_size)
    ref1_input: Layer = Input(input_size)
    bin_input_size = input_size[:2] + (bin_num,)
    bin_input: Layer = Input(bin_input_size)

    # ref1 네트워크
    ref1_l4_hc_1 = l4_hc_model_1(ref1_input)
    ref1_l4_hc_1 = UpSampling2D(8, interpolation="bilinear")(ref1_l4_hc_1)
    ref1_l4_hc_1 = Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(ref1_l4_hc_1)

    ref1_l4_hc_2 = l4_hc_model_2(ref1_input)
    ref1_l4_hc_2 = UpSampling2D(4, interpolation="bilinear")(ref1_l4_hc_2)
    ref1_l4_hc_2 = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(ref1_l4_hc_2)
    ref1_l4_hc_2 = Conv2D(
        filters * 4,
        1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(ref1_l4_hc_2)

    ref1_l4_hc_3 = l4_hc_model_3(ref1_input)
    ref1_l4_hc_3 = UpSampling2D(2, interpolation="bilinear")(ref1_l4_hc_3)
    ref1_l4_hc_3 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(ref1_l4_hc_3)
    ref1_l4_hc_3 = Conv2D(
        filters * 2,
        1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(ref1_l4_hc_3)

    ref1_l4_hc_4 = l4_hc_model_4(ref1_input)
    ref1_l4_hc_4 = Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(ref1_l4_hc_4)
    ref1_l4_hc_4 = Conv2D(
        filters, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    )(ref1_l4_hc_4)

    ref1_l4_hc_merge: Layer = concatenate(
        [ref1_l4_hc_1, ref1_l4_hc_2, ref1_l4_hc_3, ref1_l4_hc_4], axis=3
    )

    # main 네트워크
    main_l4_hc_1 = l4_hc_model_1(main_input)
    main_l4_hc_1 = UpSampling2D(8, interpolation="bilinear")(main_l4_hc_1)
    main_l4_hc_1 = Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(main_l4_hc_1)

    main_l4_hc_2 = l4_hc_model_2(main_input)
    main_l4_hc_2 = UpSampling2D(4, interpolation="bilinear")(main_l4_hc_2)
    main_l4_hc_2 = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(main_l4_hc_2)
    main_l4_hc_2 = Conv2D(
        filters * 4,
        1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(main_l4_hc_2)

    main_l4_hc_3 = l4_hc_model_3(main_input)
    main_l4_hc_3 = UpSampling2D(2, interpolation="bilinear")(main_l4_hc_3)
    main_l4_hc_3 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(main_l4_hc_3)
    main_l4_hc_3 = Conv2D(
        filters * 2,
        1,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(main_l4_hc_3)

    main_l4_hc_4 = l4_hc_model_4(main_input)
    main_l4_hc_4 = Conv2D(
        filters * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(main_l4_hc_4)
    main_l4_hc_4 = Conv2D(
        filters, 1, activation="relu", padding="same", kernel_initializer="he_normal",
    )(main_l4_hc_4)

    main_l4_hc_merge: Layer = concatenate(
        [main_l4_hc_1, main_l4_hc_2, main_l4_hc_3, main_l4_hc_4], axis=3
    )

    # 전체 합산
    total_merge: Layer = concatenate(
        [ref1_l4_hc_merge, main_l4_hc_merge, bin_input], axis=3
    )
    total_conv: Layer = Conv2D(
        100, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(total_merge)
    total_conv: Layer = Conv2D(bin_num, 1, activation="sigmoid")(total_conv)

    model: Model = Model(
        inputs=[main_input, ref1_input, bin_input], outputs=[total_conv]
    )

    model.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss=losses.binary_crossentropy,
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    if pre_trained_weights:
        model.load_weights(pre_trained_weights)

    return model
