import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils


class BgWeightedCrossentropy(LossFunctionWrapper):
    def __init__(
        self,
        bg_to_bg_weight,
        bg_to_fg_weight,
        fg_to_bg_weight,
        fg_to_fg_weight,
        from_logits=False,
        reduction=losses_utils.ReductionV2.AUTO,
        name="bg_weighted_categorical_crossentropy",
    ):
        super(BgWeightedCrossentropy, self).__init__(
            bg_weighted_crossentropy,
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


def get_bg_weights(
    y_pred,
    y_true,
    bg_to_bg_weight=1.0,
    bg_to_fg_weight=1.0,
    fg_to_bg_weight=1.0,
    fg_to_fg_weight=1.0,
):
    # predicted classes
    y_pred_classes_float = tf.one_hot(
        indices=K.argmax(y_pred), depth=tf.shape(y_pred)[-1]
    )
    # y_pred with y_true
    yp_yt = tf.concat(
        [K.expand_dims(y_pred_classes_float), K.expand_dims(y_true)], axis=-1
    )
    # yp_yt for only bg (first bin)
    yp_yt_bg = yp_yt[:, :, :, 0]

    # BG(p) to BG(t)
    cal_bg_to_bg = K.all(
        K.equal(yp_yt_bg, K.ones_like(yp_yt_bg) * [1.0, 1.0]), axis=-1, keepdims=True,
    )
    cal_bg_to_bg = K.cast(cal_bg_to_bg, K.floatx())
    # BG(p) to FG(t)
    cal_bg_to_fg = K.all(
        K.equal(yp_yt_bg, K.ones_like(yp_yt_bg) * [1.0, 0.0]), axis=-1, keepdims=True,
    )
    cal_bg_to_fg = K.cast(cal_bg_to_fg, K.floatx())
    # FG(p) to BG(t)
    cal_fg_to_bg = K.all(
        K.equal(yp_yt_bg, K.ones_like(yp_yt_bg) * [0.0, 1.0]), axis=-1, keepdims=True,
    )
    cal_fg_to_bg = K.cast(cal_fg_to_bg, K.floatx())
    # FG(p) to FG(t)
    cal_fg_to_fg = K.all(
        K.equal(yp_yt_bg, K.ones_like(yp_yt_bg) * [0.0, 0.0]), axis=-1, keepdims=True,
    )
    cal_fg_to_fg = K.cast(cal_fg_to_fg, K.floatx())

    weights = (
        bg_to_bg_weight * cal_bg_to_bg
        + bg_to_fg_weight * cal_bg_to_fg
        + fg_to_bg_weight * cal_fg_to_bg
        + fg_to_fg_weight * cal_fg_to_fg
    )

    return weights


def bg_weighted_crossentropy(
    y_true,
    y_pred,
    bg_to_bg_weight=1.0,
    bg_to_fg_weight=1.0,
    fg_to_bg_weight=1.0,
    fg_to_fg_weight=1.0,
    from_logits=False,
):
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = (
            y_true
            * K.log(y_pred)
            * get_bg_weights(
                y_pred,
                y_true,
                bg_to_bg_weight,
                bg_to_fg_weight,
                fg_to_bg_weight,
                fg_to_fg_weight,
            )
        )
        loss = -K.sum(loss, -1)
        return loss

    return loss(y_true, y_pred)
