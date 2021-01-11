from enum import Enum

from tensorflow.keras.losses import CategoricalCrossentropy

from losses.losses_weighted_cce import WeightedCrossentropy


class RefLoss(Enum):
    cce = "categorical_crossentropy"
    weighted_cce1 = "weighted_cce1"

    def get_loss(self):
        if self == RefLoss.cce:
            return CategoricalCrossentropy()
        elif self == RefLoss.weighted_cce1:
            return WeightedCrossentropy(
                weights=[
                    10.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ]
            )
        else:
            return CategoricalCrossentropy()
