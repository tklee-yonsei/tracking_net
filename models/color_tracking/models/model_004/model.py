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

    - 입력
        - (256, 256, 1)의 그레이스케일 이미지
        - (256, 256, 1)의 힌트 그레이스케일 이미지
        - (256, 256, `bin_num`)의 힌트 라벨 확률 이미지
        - (128, 128, `bin_num`)의 힌트 라벨 확률 이미지
        - (64, 64, `bin_num`)의 힌트 라벨 확률 이미지
        - (32, 32, `bin_num`)의 힌트 라벨 확률 이미지
    - 출력
        - (256, 256, `bin_num`)의 픽셀 단위 Softmax 확률

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
    bin1_input: Layer = Input(tuple(int(el / 8) for el in input_size[:2]) + (bin_num,))
    bin2_input: Layer = Input(tuple(int(el / 4) for el in input_size[:2]) + (bin_num,))
    bin3_input: Layer = Input(tuple(int(el / 2) for el in input_size[:2]) + (bin_num,))
    bin4_input: Layer = Input(input_size[:2] + (bin_num,))

    # First
    ref1_l4_1 = l4_hc_model_1(ref1_input)
    main_l4_1 = l4_hc_model_1(main_input)
    l4_merge_1: Layer = concatenate([ref1_l4_1, main_l4_1, bin1_input], axis=3)
    l4_conv_1 = Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_merge_1)
    l4_conv_1 = Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_1)
    l4_conv_up_1 = UpSampling2D(interpolation="bilinear")(l4_conv_1)
    l4_conv_up_1 = Conv2D(
        filters * 32,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_up_1)

    # Second
    ref1_l4_2 = l4_hc_model_2(ref1_input)
    main_l4_2 = l4_hc_model_2(main_input)
    l4_merge_2: Layer = concatenate(
        [l4_conv_up_1, ref1_l4_2, main_l4_2, bin2_input], axis=3
    )
    l4_conv_2 = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_merge_2)
    l4_conv_2 = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_2)
    l4_conv_up_2 = UpSampling2D(interpolation="bilinear")(l4_conv_2)
    l4_conv_up_2 = Conv2D(
        filters * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_up_2)

    # Third
    ref1_l4_3 = l4_hc_model_3(ref1_input)
    main_l4_3 = l4_hc_model_3(main_input)
    l4_merge_3: Layer = concatenate(
        [l4_conv_up_2, ref1_l4_3, main_l4_3, bin3_input], axis=3
    )
    l4_conv_3 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_merge_3)
    l4_conv_3 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_3)
    l4_conv_up_3 = UpSampling2D(interpolation="bilinear")(l4_conv_3)
    l4_conv_up_3 = Conv2D(
        filters * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(l4_conv_up_3)

    # Fourth
    ref1_l4_4 = l4_hc_model_4(ref1_input)
    main_l4_4 = l4_hc_model_4(main_input)
    l4_merge_4: Layer = concatenate(
        [l4_conv_up_3, ref1_l4_4, main_l4_4, bin4_input], axis=3
    )
    l4_conv_4 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(l4_merge_4)
    l4_conv_4 = Conv2D(
        128, 3, activation="relu", padding="same", kernel_initializer="he_normal",
    )(l4_conv_4)

    # ref1 네트워크
    total_conv: Layer = Conv2D(
        bin_num, 1, activation="relu", padding="same", kernel_initializer="he_normal"
    )(l4_conv_4)
    total_conv: Layer = Activation("softmax")(total_conv)

    model: Model = Model(
        inputs=[main_input, ref1_input, bin1_input, bin2_input, bin3_input, bin4_input],
        outputs=[total_conv],
    )

    model.compile(
        optimizer=optimizers.Adam(lr=1e-4),
        loss=losses.binary_crossentropy,
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    if pre_trained_weights:
        model.load_weights(pre_trained_weights)

    return model
