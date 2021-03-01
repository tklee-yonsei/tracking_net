from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.ops import math_ops, nn
from tensorflow.python.util.tf_export import keras_export


def cosine_proximity(y_true, y_pred, axis=-1):
    """
    Computes the cosine similarity between labels and predictions.

    Parameters
    ----------
    y_true : [type]
        The ground truth values.
    y_pred : [type]
        The prediction values.
    axis : int, optional, default=-1
        The dimension along which the cosine similarity is computed.

    Returns
    -------
    [type]
        Cosine similarity value.
    """
    y_true = nn.l2_normalize(y_true, axis=axis)
    y_pred = nn.l2_normalize(y_pred, axis=axis)
    return math_ops.reduce_sum(y_true * y_pred, axis=axis)


cosine_similarity = cosine_proximity


@keras_export("keras.metrics.CosineSimilarity")
class CosineSimilarity(MeanMetricWrapper):
    def __init__(self, name="cosine_similarity", dtype=None, axis=-1):
        """
        Computes the cosine similarity between the labels and predictions.

        `cosine similarity = (a . b) / ||a|| ||b||`

        See: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity).

        This metric keeps the average cosine similarity between `predictions` and `labels` over a stream of data.

        Parameters
        ----------
        name : str, optional, default="cosine_similarity"
            string name of the metric instance.
        dtype : [type], optional, default=None
            data type of the metric result.
        axis : int, optional, default=-1
            The dimension along which the cosine similarity is computed.

        Examples
        --------

        Standalone usage:

        >>> y_true = [[0., 1.], [1., 1.]]
        >>> y_pred = [[1., 0.], [1., 1.]]
        >>> # l2_norm(y_true) = [[0., 1.], [1./1.414], 1./1.414]]]
        >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414], 1./1.414]]]
        >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
        >>> # result = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
        >>> #        = ((0. + 0.) +  (0.5 + 0.5)) / 2
        >>> m = tf.keras.metrics.CosineSimilarity(axis=1)
        >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]])
        >>> m.result().numpy()
        0.49999997

        >>> m.reset_states()
        >>> m.update_state([[0., 1.], [1., 1.]], [[1., 0.], [1., 1.]],
        ...                sample_weight=[0.3, 0.7])
        >>> m.result().numpy()
        0.6999999

        Usage with `compile()` API:

        ```python
        model.compile(
            optimizer='sgd',
            loss='mse',
            metrics=[tf.keras.metrics.CosineSimilarity(axis=1)])
        ```
        """
        super(CosineSimilarity, self).__init__(
            cosine_similarity, name, dtype=dtype, axis=axis
        )
