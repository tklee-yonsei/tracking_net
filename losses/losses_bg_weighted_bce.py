import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils


def bce_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred + K.epsilon())
    loss += (1 - y_true) * K.log(1 - y_pred + K.epsilon())
    return K.mean(-loss, axis=-1)


def bce_loss_pointwise_weight(y_true, y_pred, weight_func):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred + K.epsilon())
    loss += (1 - y_true) * K.log(1 - y_pred + K.epsilon())
    loss *= weight_func(y_true, y_pred)
    return K.mean(-loss, axis=-1)


class BgWeightedBinaryCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        bg_to_bg_weight,
        bg_to_fg_weight,
        fg_to_bg_weight,
        fg_to_fg_weight,
        from_logits=False,
        reduction=losses_utils.ReductionV2.AUTO,
        name="bg_weighted_binary_crossentropy",
    ):
        super(BgWeightedBinaryCrossentropy, self).__init__(
            bg_weighted_binary_crossentropy,
            name=name,
            bg_to_bg_weight=bg_to_bg_weight,
            bg_to_fg_weight=bg_to_fg_weight,
            fg_to_bg_weight=fg_to_bg_weight,
            fg_to_fg_weight=fg_to_fg_weight,
            reduction=reduction,
            from_logits=from_logits,
        )
        self.bg_to_bg_weight = bg_to_bg_weight
        self.bg_to_fg_weight = bg_to_fg_weight
        self.fg_to_bg_weight = fg_to_bg_weight
        self.fg_to_fg_weight = fg_to_fg_weight
        self.from_logits = from_logits


def softargmax(x, beta=1e10):
    x = tf.convert_to_tensor(x)
    x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1)


def get_bg_weights(
    y_true,
    y_pred,
    bg_to_bg_weight=1.0,
    bg_to_fg_weight=1.0,
    fg_to_bg_weight=1.0,
    fg_to_fg_weight=1.0,
):
    y_pred_classes_float = tf.cast(tf.cast(y_pred + 0.5, tf.int32), tf.float32)
    yt_yp = tf.concat(
        [K.expand_dims(y_true), K.expand_dims(y_pred_classes_float)], axis=-1
    )

    # BG(t) to BG(p)
    cal_bg_to_bg = K.all(
        K.equal(yt_yp, K.ones_like(yt_yp) * [0.0, 0.0]), axis=-1, keepdims=True,
    )
    cal_bg_to_bg = K.cast(cal_bg_to_bg, K.floatx())
    # BG(t) to FG(p)
    cal_bg_to_fg = K.all(
        K.equal(yt_yp, K.ones_like(yt_yp) * [0.0, 1.0]), axis=-1, keepdims=True,
    )
    cal_bg_to_fg = K.cast(cal_bg_to_fg, K.floatx())
    # FG(t) to BG(p)
    cal_fg_to_bg = K.all(
        K.equal(yt_yp, K.ones_like(yt_yp) * [1.0, 0.0]), axis=-1, keepdims=True,
    )
    cal_fg_to_bg = K.cast(cal_fg_to_bg, K.floatx())
    # FG(t) to FG(p)
    cal_fg_to_fg = K.all(
        K.equal(yt_yp, K.ones_like(yt_yp) * [1.0, 1.0]), axis=-1, keepdims=True,
    )
    cal_fg_to_fg = K.cast(cal_fg_to_fg, K.floatx())

    weights = (
        bg_to_bg_weight * cal_bg_to_bg
        + bg_to_fg_weight * cal_bg_to_fg
        + fg_to_bg_weight * cal_fg_to_bg
        + fg_to_fg_weight * cal_fg_to_fg
    )
    weights = tf.squeeze(weights, axis=-1)
    return weights


def bg_weighted_binary_crossentropy(
    y_true,
    y_pred,
    bg_to_bg_weight=1.0,
    bg_to_fg_weight=1.0,
    fg_to_bg_weight=1.0,
    fg_to_fg_weight=1.0,
    from_logits=False,
):
    weight_func = lambda y_true, y_pred: get_bg_weights(
        y_true,
        y_pred,
        bg_to_bg_weight,
        bg_to_fg_weight,
        fg_to_bg_weight,
        fg_to_fg_weight,
    )
    return bce_loss_pointwise_weight(y_true, y_pred, weight_func)
