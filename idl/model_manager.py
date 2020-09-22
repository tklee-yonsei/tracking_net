from typing import List, Optional, Tuple

import keras
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

    def get_input_sizes(self) -> List[Tuple[int, int]]:
        return list(map(lambda el: (el[1][0], el[1][1]), self.inputs))

    def get_output_sizes(self) -> List[Tuple[int, int]]:
        return list(map(lambda el: (el[1][0], el[1][1]), self.outputs))


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


class ModelHelper:
    def __init__(self, model_descriptor: ModelDescriptor, alpha: float = 1.0):
        self.model_descriptor = model_descriptor
        self.alpha = alpha

    def get_model(self) -> Model:
        raise NotImplementedError

    def compile_model(
        self,
        model: Model,
        optimizer: Optimizer,
        loss_list: List[LossDescriptor],
        metrics: List[Metric] = [],
        model_descriptor: Optional[ModelDescriptor] = None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        **kwargs
    ) -> Model:
        new_model = keras.models.clone_model(model)
        _model_descriptor = (
            self.model_descriptor if model_descriptor is None else model_descriptor
        )
        compile_model(
            new_model,
            model_descriptor=_model_descriptor,
            optimizer=optimizer,
            loss_list=loss_list,
            metrics=metrics,
            sample_weight_mode=sample_weight_mode,
            weighted_metrics=weighted_metrics,
            target_tensors=target_tensors,
            **kwargs
        )
        return new_model
