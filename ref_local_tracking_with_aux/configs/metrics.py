from enum import Enum

from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.metrics import BinaryAccuracy


class RefMetric(Enum):
    none = "none"
    ca = "categorical_accuracy"
    ba = "binary_accuracy"

    def get_metric(self):
        if self == RefMetric.ca:
            return CategoricalAccuracy(name="categorical_accuracy")
        if self == RefMetric.ba:
            return BinaryAccuracy(name="binary_accuracy")
        else:
            return None

    @staticmethod
    def get_default() -> str:
        return RefMetric.none.value
