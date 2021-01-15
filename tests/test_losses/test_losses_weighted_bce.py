from unittest import TestCase

import tensorflow as tf
from keras import backend as K
from losses.losses_bg_weighted_bce import (
    bce_loss,
    bg_weighted_binary_crossentropy,
    get_bg_weights,
)


class TestWeightedBCE(TestCase):
    def test_bce_loss(self):
        # data
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = tf.constant(y_true, dtype=tf.float32)

        y_true2 = [[[0, 1], [0, 0]], [[1, 0], [0, 1]]]
        y_pred2 = [[[0.6, 0.4], [0.4, 0.6]], [[0.3, 0.7], [0.8, 0.2]]]
        y_true2 = tf.constant(y_true2, dtype=tf.float32)
        y_pred2 = tf.constant(y_true2, dtype=tf.float32)

        # target
        target = bce_loss(y_true, y_pred)
        target2 = bce_loss(y_true2, y_pred2)

        # calculate
        result = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        result2 = tf.keras.losses.binary_crossentropy(y_true2, y_pred2)

        # assertion
        tf.debugging.assert_equal(target, result)
        tf.debugging.assert_equal(target2, result2)

    def test_get_bg_weights(self):
        # data
        y_true2 = [[[0, 1], [0, 0]], [[1, 0], [0, 1]]]
        y_pred2 = [[[0.6, 0.4], [0.4, 0.6]], [[0.3, 0.7], [0.8, 0.6]]]
        y_true2 = tf.constant(y_true2, dtype=tf.float32)
        y_pred2 = tf.constant(y_pred2, dtype=tf.float32)

        # target
        target = tf.constant(
            [[[[5.0], [10.0]], [[1.0], [5.0]]], [[[10.0], [5.0]], [[5.0], [2.0]]]]
        )

        # calculate
        result = get_bg_weights(y_true2, y_pred2, 1, 5, 10, 2)

        # assertion
        tf.debugging.assert_equal(target, result)

    def test_bg_weighted_binary_crossentropy(self):
        # data
        y_true2 = [[[[0, 1], [0, 0]], [[1, 0], [0, 1]]]]
        y_pred2 = [[[[0.6, 0.4], [0.4, 0.6]], [[0.3, 0.7], [0.8, 0.6]]]]
        y_true2 = tf.constant(y_true2, dtype=tf.float32)
        y_pred2 = tf.constant(y_pred2, dtype=tf.float32)

        # target
        target = tf.keras.losses.binary_crossentropy(y_true2, y_pred2)

        # calculate
        result = bg_weighted_binary_crossentropy(y_true2, y_pred2, 1, 1, 1, 1)
        result2 = bg_weighted_binary_crossentropy(y_true2, y_pred2, 1, 5, 10, 2)

        # assertion
        tf.debugging.assert_equal(target, result)
