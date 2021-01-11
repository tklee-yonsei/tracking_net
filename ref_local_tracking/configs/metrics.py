from enum import Enum

from tensorflow.keras.metrics import CategoricalAccuracy


class RefMetric(Enum):
    ca = "categorical_accuracy"

    def get_metric(self):
        if self == RefMetric.ca:
            return CategoricalAccuracy(name="accuracy")
        else:
            return CategoricalAccuracy(name="accuracy")
