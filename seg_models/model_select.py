from enum import Enum, unique
from typing import Tuple

from common_py import ArgTypeMixin
from keras import Model


@unique
class ModelSelect(ArgTypeMixin, Enum):
    unet_l1 = "unet_l1"
    unet_l4 = "unet_l4"
    unet_l4_2 = "unet_l4_2"
    unet_l4_m = "unet_l4_m"
    vanilla_unet = "vanilla_unet"

    def get_model(
        self,
        input_size: Tuple[int, int, int],
        pre_trained_weight_path: str = None,
        filters: int = 16,
    ) -> Model:
        if self == ModelSelect.unet_l1:
            from seg_models.unet_l1.unet_l1 import unet_l1

            return unet_l1(
                filters=filters,
                pre_trained_weight_path=pre_trained_weight_path,
                input_size=input_size,
            )
        elif self == ModelSelect.unet_l4:
            from seg_models.unet_l4.unet_l4 import unet_l4

            return unet_l4(
                filters=filters,
                pre_trained_weight_path=pre_trained_weight_path,
                input_size=input_size,
            )
        elif self == ModelSelect.unet_l4_2:
            from seg_models.unet_l4.unet_l4 import unet_l4

            return unet_l4(
                filters=filters,
                pre_trained_weight_path=pre_trained_weight_path,
                input_size=input_size,
            )
        elif self == ModelSelect.unet_l4_m:
            from seg_models.unet_l4.unet_l4 import unet_l4

            m = unet_l4(pre_trained_weight_path=pre_trained_weight_path)
            mm = Model(inputs=m.input, outputs=m.get_layer(m.layers[8].name).output)
            return mm
        else:
            from seg_models.vanilla_unet.vanilla_unet import vanilla_unet

            return vanilla_unet(
                pre_trained_weight_path=pre_trained_weight_path, input_size=input_size
            )
