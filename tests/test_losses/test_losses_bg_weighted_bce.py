from unittest import TestCase

import tensorflow as tf
from keras import backend as K
from losses.losses_bg_weighted_bce import (
    bce_loss,
    bg_weighted_binary_crossentropy,
    get_bg_weights,
)


class TestBGWeightedBCE(TestCase):
    def test_bce_loss(self):
        # data
        y_true = [[0, 1], [0, 0]]
        y_pred = [[0.6, 0.4], [0.4, 0.6]]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = tf.constant(y_true, dtype=tf.float32)

        y_true2 = [[[[0], [1]], [[0], [0]]], [[[1], [0]], [[0], [1]]]]
        y_pred2 = [[[[0.6], [0.4]], [[0.4], [0.6]]], [[[0.3], [0.7]], [[0.8], [0.6]]]]
        y_true2 = tf.constant(y_true2, dtype=tf.float32)
        y_pred2 = tf.constant(y_true2, dtype=tf.float32)

        # target
        target = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        target2 = tf.keras.losses.binary_crossentropy(y_true2, y_pred2)

        # calculate
        result = bce_loss(y_true, y_pred)
        result2 = bce_loss(y_true2, y_pred2)

        # assertion
        tf.debugging.assert_equal(target, result)
        tf.debugging.assert_equal(target2, result2)

    def test_get_bg_weights(self):
        # data
        2, 2, 2, 1
        y_true2 = [[[[0], [1]], [[0], [0]]], [[[1], [0]], [[0], [1]]]]
        y_pred2 = [[[[0.6], [0.4]], [[0.4], [0.6]]], [[[0.3], [0.7]], [[0.8], [0.6]]]]
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

    def test_bg_weighted_binary_crossentropy_1(self):
        # data
        y_true2 = [[[[0], [1]], [[0], [0]]], [[[1], [0]], [[0], [1]]]]
        y_pred2 = [[[[0.6], [0.4]], [[0.4], [0.6]]], [[[0.3], [0.7]], [[0.8], [0.6]]]]
        y_true2 = tf.constant(y_true2, dtype=tf.float32)
        y_pred2 = tf.constant(y_pred2, dtype=tf.float32)

        # target
        target = tf.keras.losses.binary_crossentropy(y_true2, y_pred2)

        # calculate
        result = bg_weighted_binary_crossentropy(y_true2, y_pred2, 1, 1, 1, 1)

        # assertion
        tf.debugging.assert_equal(target, result)

    def test_bg_weighted_binary_crossentropy_2(self):
        # data
        y_true = [[[0, 1], [0, 0]], [[1, 0], [0, 1]]]
        y_pred = [[[0.6, 0.4], [0.4, 0.6]], [[0.3, 0.7], [0.8, 0.6]]]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        y_true2 = [[[[0], [1]], [[0], [0]]], [[[1], [0]], [[0], [1]]]]
        y_pred2 = [[[[0.6], [0.4]], [[0.4], [0.6]]], [[[0.3], [0.7]], [[0.8], [0.6]]]]
        y_true2 = tf.constant(y_true2, dtype=tf.float32)
        y_pred2 = tf.constant(y_pred2, dtype=tf.float32)

        # target
        target = tf.constant([[[6.872179, 2.5461392], [9.029793, 4.534419]]])
        target2 = tf.constant(
            [
                [[0.9162906, 0.9162905], [0.5108254, 0.9162906]],
                [[1.2039725, 1.2039725], [1.6094375, 0.5108254]],
            ]
        )
        weight = tf.constant([[[5.0, 10.0], [1.0, 5.0]], [[10.0, 5.0], [5.0, 2.0]]])
        target2 *= weight

        # calculate
        result = bg_weighted_binary_crossentropy(y_true, y_pred, 1, 5, 10, 2)
        result2 = bg_weighted_binary_crossentropy(y_true2, y_pred2, 1, 5, 10, 2)

        # assertion
        tf.debugging.assert_equal(target, result)
        tf.debugging.assert_equal(target2, result2)
