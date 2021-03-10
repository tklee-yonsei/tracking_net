from tensorflow.python.framework import ops, smart_cond
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.metrics.categorical_crossentropy", "keras.losses.categorical_crossentropy"
)
@dispatch.add_dispatch_support
def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    """
    Computes the categorical crossentropy loss.

    Parameters
    ----------
    y_true : [type]
        Tensor of one-hot true targets.
    y_pred : [type]
        Tensor of predicted targets.
    from_logits : bool, optional, default=False
        Whether `y_pred` is expected to be a logits tensor. 
        By default, we assume that `y_pred` encodes a probability distribution.
    label_smoothing : float, optional, default=0
        Float in [0, 1]. If > `0` then smooth the labels.

    Returns
    -------
    [type]
        Categorical crossentropy loss value.

    Examples
    --------
    Standalone usage:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.0513, 2.303], dtype=float32)
    """
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    label_smoothing = ops.convert_to_tensor_v2_with_dispatch(
        label_smoothing, dtype=K.floatx()
    )

    def _smooth_labels():
        num_classes = math_ops.cast(array_ops.shape(y_true)[-1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


@keras_export("keras.losses.CategoricalCrossentropy")
class CategoricalCrossentropy(LossFunctionWrapper):
    """
    Computes the crossentropy loss between the labels and predictions.

    Use this crossentropy loss function when there are two or more label classes.
    We expect labels to be provided in a `one_hot` representation. If you want to
    provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
    There should be `# classes` floating point values per feature.

    In the snippet below, there is `# classes` floating pointing values per
    example. The shape of both `y_pred` and `y_true` are
    `[batch_size, num_classes]`.

    Examples
    --------
    Standalone usage:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> # Using 'auto'/'sum_over_batch_size' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy()
    >>> cce(y_true, y_pred).numpy()
    1.177

    >>> # Calling with 'sample_weight'.
    >>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
    0.814

    >>> # Using 'sum' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.SUM)
    >>> cce(y_true, y_pred).numpy()
    2.354

    >>> # Using 'none' reduction type.
    >>> cce = tf.keras.losses.CategoricalCrossentropy(
    ...     reduction=tf.keras.losses.Reduction.NONE)
    >>> cce(y_true, y_pred).numpy()
    array([0.0513, 2.303], dtype=float32)

    Usage with the `compile()` API:

    ```python
    model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
    ```
    """

    def __init__(
        self,
        from_logits=False,
        label_smoothing=0,
        reduction=losses_utils.ReductionV2.AUTO,
        name="categorical_crossentropy",
    ):
        """
        Initializes `CategoricalCrossentropy` instance.

        Parameters
        ----------
        from_logits : bool, optional, default=False
            Whether `y_pred` is expected to be a logits tensor. 
            By default, we assume that `y_pred` encodes a probability distribution.
            **Note - Using from_logits=True is more numerically stable.**
        label_smoothing : float, optional, default=0
            Float in [0, 1]. When > 0, label values are smoothed, meaning the confidence on label values are relaxed. 
            e.g. `label_smoothing=0.2` means that we will use a value of `0.1` for label `0` and `0.9` for label `1`"
        reduction : [type], optional, default=losses_utils.ReductionV2.AUTO
            Type of `tf.keras.losses.Reduction` to apply to loss. 
            Default value is `AUTO`. `AUTO` indicates that the reduction option will be determined by the usage context. 
            For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`. 
            When used with `tf.distribute.Strategy`, outside of built-in training loops such as `tf.keras` `compile` 
            and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. 
            Please see this custom training [tutorial](https://www.tensorflow.org/tutorials/distribute/custom_training) for more details.
        name : str, optional, default="categorical_crossentropy"
            Optional name for the op.
        """
        super(CategoricalCrossentropy, self).__init__(
            categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        )
