from typing import List, Optional, Tuple

from keras.losses import Loss
from keras.metrics import Metric
from keras.models import Model
from keras.optimizers import Optimizer


class LossDescriptor:
    """
    손실 설명을 위한 도구입니다.
    """

    def __init__(self, loss: Loss, weight: int = 1.0):
        self.loss = loss
        self.weight = weight


class ModelDescriptor:
    """
    모델 설명을 위한 도구입니다. 

    - 모델에 해당되는 입력의 이름, 입력의 shape을 지정합니다.
    - 모델에 해당되는 출력의 이름, 출력의 shape을 지정합니다.
    """

    def __init__(
        self,
        inputs: List[Tuple[str, Tuple[int, int, int]]],
        outputs: List[Tuple[str, Tuple[int, int, int]]],
    ):
        self.inputs = inputs
        self.outputs = outputs

    def get_input_sizes(self) -> List[Tuple[int, int]]:
        """
        모델의 입력의 세로, 가로 크기를 반환합니다.

        Returns
        -------
        List[Tuple[int, int]]
            이미지의 세로, 가로 크기 리스트
        """
        return list(map(lambda el: (el[1][0], el[1][1]), self.inputs))

    def get_output_sizes(self) -> List[Tuple[int, int]]:
        """
        모델의 출력의 세로, 가로 크기를 반환합니다.

        Returns
        -------
        List[Tuple[int, int]]
            이미지의 세로, 가로 크기 리스트
        """
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
    """
    모델을 컴파일합니다.

    Parameters
    ----------
    model : Model
        컴파일 할 모델
    model_descriptor : ModelDescriptor
        모델 설명 도구
    optimizer : Optimizer
        옵티마이저
    loss_list : List[LossDescriptor]
        손실 설명 도구의 리스트
    metrics : List[Metric]
        메트릭 리스트
    sample_weight_mode : [type], optional
        `sample_weight_mode`, by default None
    weighted_metrics : [type], optional
        `weighted_metrics`, by default None
    target_tensors : [type], optional
        `target_tensors`, by default None
    """
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
    """
    [Interface]
    """

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
        _model_descriptor = model_descriptor
        if model_descriptor is None:
            _model_descriptor = self.model_descriptor
        compile_model(
            model,
            model_descriptor=_model_descriptor,
            optimizer=optimizer,
            loss_list=loss_list,
            metrics=metrics,
            sample_weight_mode=sample_weight_mode,
            weighted_metrics=weighted_metrics,
            target_tensors=target_tensors,
            **kwargs
        )
        return model
