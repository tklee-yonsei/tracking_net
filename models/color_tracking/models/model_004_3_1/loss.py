import tensorflow as tf


import keras


class CustomCategoricalCrossentropy(keras.losses.LossFunctionWrapper):
    def __init__(
        self,
        from_logits=False,
        label_smoothing=0,
        reduction=keras.losses.losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
        name="categorical_crossentropy_t2",
    ):
        super(CustomCategoricalCrossentropy, self).__init__(
            categorical_crossentropy_t2,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
        )


import keras.backend as K


def categorical_crossentropy_t(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    value = K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

    print("value: {}".format(value))
    print("-------------")

    return value


def categorical_crossentropy_t2(y_true, y_pred, from_logits=False, label_smoothing=0):
    print("categorical_crossentropy_t2")

    y_pred = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -y_true * K.log(y_pred)
    loss = K.sum(loss, -1)

    loss = loss + 10.0

    return loss

