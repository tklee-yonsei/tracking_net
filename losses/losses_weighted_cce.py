import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper, categorical_crossentropy
from tensorflow.python.keras.utils import losses_utils


class WeightedCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        weights,
        from_logits=False,
        reduction=losses_utils.ReductionV2.AUTO,
        name="weighted_categorical_crossentropy",
    ):
        super(WeightedCrossentropy, self).__init__(
            weighted_crossentropy,
            name=name,
            weights=weights,
            reduction=reduction,
            from_logits=from_logits,
        )
        self.weights = weights
        self.from_logits = from_logits


def weighted_crossentropy(y_true, y_pred, weights, from_logits=False):
    """
    [summary]

    >>> weights = [10., 1., 1.]
    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, weights)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()

    Parameters
    ----------
    y_true : [type]
        [description]
    y_pred : [type]
        [description]
    from_logits : bool, optional
        [description], by default False
    label_smoothing : int, optional
        [description], by default 0
    range : int, optional
        [description], by default 0
    max : float, optional
        [description], by default 1.0

    Returns
    -------
    [type]
        [description]

    Examples
    --------
    >>> from image_keras.custom.losses_binary_boundary_crossentropy import binary_boundary_crossentropy
    >>> import cv2
    >>> a = cv2.imread("tests/test_resources/a.png", cv2.IMREAD_GRAYSCALE)
    >>> a_modified = (a / 255).reshape(1, a.shape[0], a.shape[1], 1)
    >>> binary_boundary_crossentropy(a_modified, a_modified, range=1, max=2)
    """
    weights = K.constant(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss(y_true, y_pred)
