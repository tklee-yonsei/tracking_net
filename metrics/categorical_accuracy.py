from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.ops import math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.metrics.categorical_accuracy")
@dispatch.add_dispatch_support
def categorical_accuracy(y_true, y_pred):
    """
    Calculates how often predictions matches one-hot labels.

    Parameters
    ----------
    y_true : [type]
        One-hot ground truth values.
    y_pred : [type]
        The prediction values.

    Returns
    -------
    [type]
        Categorical accuracy values.

    Examples
    --------

    Standalone usage:
    >>> y_true = [[0, 0, 1], [0, 1, 0]]
    >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> m = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    >>> assert m.shape == (2,)
    >>> m.numpy()
    array([0., 1.], dtype=float32)

    You can provide logits of classes as `y_pred`, since argmax of logits and probabilities are same.
    """
    return math_ops.cast(
        math_ops.equal(
            math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)
        ),
        K.floatx(),
    )


@keras_export("keras.metrics.CategoricalAccuracy")
class CategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name="categorical_accuracy", dtype=None):
        """
        Calculates how often predictions matches one-hot labels.

        You can provide logits of classes as `y_pred`, since argmax of
        logits and probabilities are same.

        This metric creates two local variables, `total` and `count` that are used to
        compute the frequency with which `y_pred` matches `y_true`. This frequency is
        ultimately returned as `categorical accuracy`: an idempotent operation that
        simply divides `total` by `count`.

        `y_pred` and `y_true` should be passed in as vectors of probabilities, rather
        than as labels. If necessary, use `tf.one_hot` to expand `y_true` as a vector.

        If `sample_weight` is `None`, weights default to 1.
        Use `sample_weight` of 0 to mask values.

        Parameters
        ----------
        name : str, optional, default="categorical_accuracy"
            string name of the metric instance.
        dtype : [type], optional, default=None
            data type of the metric result.

        Examples
        --------
        Standalone usage:

        >>> m = tf.keras.metrics.CategoricalAccuracy()
        >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
        ...                 [0.05, 0.95, 0]])
        >>> m.result().numpy()
        0.5

        >>> m.reset_states()
        >>> m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8],
        ...                 [0.05, 0.95, 0]],
        ...                sample_weight=[0.7, 0.3])
        >>> m.result().numpy()
        0.3

        Usage with `compile()` API:

        ```python
        model.compile(
            optimizer='sgd',
            loss='mse',
            metrics=[tf.keras.metrics.CategoricalAccuracy()])
        ```
        """
        super(CategoricalAccuracy, self).__init__(
            categorical_accuracy, name, dtype=dtype
        )
