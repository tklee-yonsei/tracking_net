from typing import List, Tuple

from keras.losses import Loss
from keras.metrics import Metric
from keras.models import Model
from keras.optimizers import Optimizer


class LossDescriptor:
    def __init__(self, loss: Loss, weight: int = 1.0):
        self.loss = loss
        self.weight = weight


class ModelDescriptor:
    def __init__(
        self,
        inputs: List[Tuple[str, Tuple[int, int, int]]],
        outputs: List[Tuple[str, Tuple[int, int, int]]],
    ):
        self.inputs = inputs
        self.outputs = outputs


def compile_model(
    model: Model,
    model_descriptor: ModelDescriptor,
    optimizer: Optimizer,
    loss_list: List[LossDescriptor],
    metrics: List[Metric],
    sample_weight_mode=None,
    weighted_metrics=None,
    target_tensors=None,
    **kwargs
):
    losses = dict(
        zip(
            list(map(lambda el: el[0], model_descriptor.outputs)),
            list(map(lambda el: el.loss, loss_list)),
        )
    )
    loss_weights = dict(
        zip(
            list(map(lambda el: el[0], model_descriptor.outputs)),
            list(map(lambda el: el.weight, loss_list)),
        )
    )
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
        sample_weight_mode=sample_weight_mode,
        target_tensors=target_tensors,
        **kwargs
    )
