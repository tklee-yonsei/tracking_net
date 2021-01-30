from enum import Enum

from tensorflow.keras.optimizers import Adam


class RefOptimizer(Enum):
    adam1 = "adam1"

    def get_optimizer(self):
        if self == RefOptimizer.adam1:
            return Adam(lr=1e-4)
        else:
            return Adam(lr=1e-4)

    @staticmethod
    def get_default() -> str:
        return RefOptimizer.adam1.value
