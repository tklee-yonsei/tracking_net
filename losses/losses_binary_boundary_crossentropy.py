import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper, binary_crossentropy
from tensorflow.python.keras.utils import losses_utils


def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


class BinaryBoundaryCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0,
        reduction=losses_utils.ReductionV2.AUTO,
        range=1,
        max=2.0,
        name="binary_boundary_crossentropy",
    ):
        super(BinaryBoundaryCrossentropy, self).__init__(
            binary_boundary_crossentropy,
            name=name,
            reduction=reduction,
            range=range,
            max=max,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        )
        self.from_logits = from_logits
        self.range = range
        self.max = max


def binary_boundary_crossentropy(
    y_true,
    y_pred,
    from_logits=False,
    label_smoothing=0,
    range: int = 1,
    max: float = 2.0,
):
    """
    [summary]

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
    bce = binary_crossentropy(
        y_true=y_true,
        y_pred=y_pred,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
    )

    def count_around_blocks(arr, range: int = 1):
        ones = tf.ones_like(arr)
        size = range * 2 + 1
        if range < 1:
            size = 1
        extracted = tf.image.extract_patches(
            images=ones,
            sizes=[1, size, size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        result = tf.reduce_sum(extracted, axis=-1)
        if range > 0:
            result -= 1
        return result

    def count_around_blocks2(arr, range: int = 1):
        size = range * 2 + 1
        if range < 1:
            size = 1
        extracted = tf.image.extract_patches(
            images=arr,
            sizes=[1, size, size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        e_base = extracted[:, :, :, tf.shape(extracted)[-1] // 2]
        e_base = tf.reshape(e_base, (-1, tf.shape(arr)[1], tf.shape(arr)[2], 1))
        e_base_expanded = tf.reshape(
            tf.repeat(e_base, tf.shape(extracted)[-1]),
            (-1, tf.shape(arr)[1], tf.shape(arr)[2], tf.shape(extracted)[-1]),
        )
        same_values = tf.math.equal(extracted, e_base_expanded)
        result_1 = tf.shape(extracted)[-1] - tf.cast(
            tf.math.count_nonzero(same_values, axis=-1), tf.int32
        )
        result_1 = tf.reshape(result_1, (-1, tf.shape(arr)[1], tf.shape(arr)[2], 1))
        result_1 = tf.cast(result_1, tf.float32)
        result_1 += arr
        block_counts = tf.reshape(
            count_around_blocks(arr, range), (-1, tf.shape(arr)[1], tf.shape(arr)[2], 1)
        )
        modify_result_1 = -(size ** 2 - block_counts)
        modify_result_1 = modify_result_1 * arr
        modify_result_1 = tf.cast(modify_result_1, tf.float32)
        diff_block_count = result_1 + modify_result_1
        return diff_block_count

    around_block_count = count_around_blocks(y_true, range=range)
    around_block_count = tf.reshape(
        around_block_count, (-1, tf.shape(y_true)[1], tf.shape(y_true)[2], 1)
    )
    around_block_count = tf.cast(around_block_count, tf.float32)

    diff_block_count = count_around_blocks2(y_true, range=range)
    diff_block_count = tf.reshape(
        diff_block_count, (-1, tf.shape(y_true)[1], tf.shape(y_true)[2], 1)
    )
    diff_block_count = tf.cast(diff_block_count, tf.float32)

    diff_ratio = diff_block_count / around_block_count
    diff_ratio = 1.0 + tf.cast(tf.math.maximum(max - 1.0, 0), tf.float32) * diff_ratio
    diff_ratio = tf.reshape(diff_ratio, (-1, tf.shape(y_true)[1], tf.shape(y_true)[2]))

    return bce * diff_ratio
