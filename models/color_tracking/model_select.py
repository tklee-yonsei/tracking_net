from enum import Enum, unique
from typing import Tuple

from common_py import ArgTypeMixin
from keras import Model


@unique
class ModelSelect(ArgTypeMixin, Enum):
    color_tracking_model_001 = "color_tracking_model_001"
    color_tracking_model_002 = "color_tracking_model_002"
    color_tracking_model_004 = "color_tracking_model_004"
    color_tracking_model_004_1 = "color_tracking_model_004_1"
    color_tracking_model_004_3 = "color_tracking_model_004_3"
    color_tracking_model_004_3_1 = "color_tracking_model_004_3_1"
    color_tracking_model_004_3_4 = "color_tracking_model_004_3_4"
    color_tracking_model_005 = "color_tracking_model_005"
    color_tracking_model_005_1 = "color_tracking_model_005_1"
    color_tracking_model_006 = "color_tracking_model_006"
    color_tracking_model_007 = "color_tracking_model_007"
    color_tracking_model_007_1 = "color_tracking_model_007_1"

    def get_model(
        self,
        input_size: Tuple[int, int, int],
        unet_pre_trained_weights: str,
        pre_trained_weight_path: str = None,
        bin_num: int = 30,
    ) -> Model:
        if self == ModelSelect.color_tracking_model_002:
            from color_tracking.models.model_002.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_004:
            from color_tracking.models.model_004.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_004_1:
            from color_tracking.models.model_004_1.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_004_3:
            from color_tracking.models.model_004_3.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_004_3_1:
            from color_tracking.models.model_004_3_1.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_004_3_4:
            from color_tracking.models.model_004_3_4.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_005:
            from color_tracking.models.model_005.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_005_1:
            from color_tracking.models.model_005_1.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_006:
            from color_tracking.models.model_006.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_007:
            from color_tracking.models.model_007.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        elif self == ModelSelect.color_tracking_model_007_1:
            from color_tracking.models.model_007_1.model import model

            print(ModelSelect.color_tracking_model_007_1)

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
        else:
            from color_tracking.models.model_001.model import model

            return model(
                unet_pre_trained_weights=unet_pre_trained_weights,
                bin_num=bin_num,
                input_size=input_size,
                pre_trained_weights=pre_trained_weight_path,
            )
