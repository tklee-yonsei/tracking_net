import keras.backend as K
import tensorflow as tf
from keras.metrics import MeanMetricWrapper


class BinaryClassMeanIoU(MeanMetricWrapper):
    def __init__(self, name="mean_iou", dtype=None, threshold=0.5):
        super(BinaryClassMeanIoU, self).__init__(
            binary_class_mean_iou, name, dtype=dtype, threshold=threshold
        )


def binary_class_mean_iou(y_true, y_pred, threshold=0.5):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > threshold, "float32")
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, "float32"))
    return iou
