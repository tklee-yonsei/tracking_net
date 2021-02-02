from enum import Enum

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.losses import BinaryCrossentropy


class RefLoss(Enum):
    cce = "categorical_crossentropy"
    bce = "binary_crossentropy"

    def get_loss(self):
        if self == RefLoss.cce:
            return CategoricalCrossentropy()
        elif self == RefLoss.bce:
            return BinaryCrossentropy()
        else:
            return CategoricalCrossentropy()

    @staticmethod
    def get_default() -> str:
        return RefLoss.cce.value
