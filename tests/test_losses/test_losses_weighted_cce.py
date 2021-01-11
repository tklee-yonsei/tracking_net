from unittest import TestCase

import tensorflow as tf
from keras import backend as K
from losses.losses_weighted_cce import weighted_crossentropy


class TestWeightedCCE(TestCase):
    def test_weighted_crossentropy_equal_weight_1(self):
        # data
        weights = [1.0, 1.0, 1.0]
        y_true = [[0, 1, 0], [0, 0, 1]]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

        # target
        target = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # calculate
        result = weighted_crossentropy(y_true, y_pred, weights)

        # assertion
        self.assertTrue(tf.math.reduce_all(tf.math.equal(result, target)))

    def test_weighted_crossentropy(self):
        # data
        weights = [1.0, 2.0, 1.0]
        y_true = [[0, 1, 0], [0, 0, 1]]
        y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

        # target as K
        sum_of_y_pred = K.sum(y_pred, axis=-1, keepdims=True)
        mean_y_pred = y_pred / sum_of_y_pred
        cliped_y_pred = K.clip(mean_y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(cliped_y_pred) * weights
        sum_loss = -K.sum(loss, axis=-1)
        k_target = sum_loss

        # target as tf
        sum_of_y_pred = tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        mean_y_pred = y_pred / sum_of_y_pred
        cliped_y_pred = tf.clip_by_value(mean_y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * tf.math.log(cliped_y_pred) * weights
        sum_loss = -tf.reduce_sum(loss, axis=-1)
        tf_target = sum_loss

        # calculate
        result = weighted_crossentropy(y_true, y_pred, weights)

        # assertion
        self.assertTrue(tf.math.reduce_all(tf.math.equal(k_target, tf_target)))
        self.assertTrue(tf.math.reduce_all(tf.math.equal(result, k_target)))
