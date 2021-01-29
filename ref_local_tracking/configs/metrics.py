from enum import Enum

from tensorflow.keras.metrics import CategoricalAccuracy


class RefMetric(Enum):
    none = "none"
    ca = "categorical_accuracy"

    def get_metric(self):
        if self == RefMetric.ca:
            return CategoricalAccuracy(name="accuracy")
        else:
            return None

    @staticmethod
    def get_default() -> str:
        return RefMetric.none.value
