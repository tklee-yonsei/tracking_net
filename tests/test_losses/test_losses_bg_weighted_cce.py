from unittest import TestCase

import tensorflow as tf
from losses.losses_bg_weighted_cce import (
    bg_weighted_categorical_crossentropy,
    cce_loss,
    get_bg_weights,
)


class TestBGWeightedCCE(TestCase):
    def test_cce_loss(self):
        # data
        y_true = [
            [[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]],
            [[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
        ]
        y_pred = [
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
            [[[0.5, 0.2, 0.3], [0.1, 0.7, 0.2]], [[0.55, 0.25, 0.3], [0.6, 0.1, 0.3]]],
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
        ]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        # target
        target = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # calculate
        result = cce_loss(y_true, y_pred)

        # assertion
        tf.debugging.assert_equal(target, result)

    def test_get_bg_weights(self):
        # data
        y_true = [
            [[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]],
            [[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
        ]
        y_pred = [
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
            [[[0.5, 0.2, 0.3], [0.1, 0.7, 0.2]], [[0.55, 0.25, 0.3], [0.6, 0.1, 0.3]]],
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
        ]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        # target
        target = tf.constant(
            [
                [[[2.0], [10.0]], [[5.0], [2.0]]],
                [[[1.0], [2.0]], [[5.0], [1.0]]],
                [[[2.0], [10.0]], [[5.0], [2.0]]],
            ]
        )

        # calculate
        weights = get_bg_weights(
            y_true,
            y_pred,
            bg_to_bg_weight=1,
            bg_to_fg_weight=5,
            fg_to_bg_weight=10,
            fg_to_fg_weight=2,
        )

        # assertion
        tf.debugging.assert_equal(target, weights)

    def test_bg_weighted_categorical_crossentropy_1(self):
        # data
        y_true = [
            [[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]],
            [[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
        ]
        y_pred = [
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
            [[[0.5, 0.2, 0.3], [0.1, 0.7, 0.2]], [[0.55, 0.25, 0.3], [0.6, 0.1, 0.3]]],
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
        ]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        # target
        target = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # calculate
        result = bg_weighted_categorical_crossentropy(y_true, y_pred, 1, 1, 1, 1)

        # assertion
        tf.debugging.assert_equal(target, result)

    def test_bg_weighted_categorical_crossentropy_2(self):
        # data
        y_true = [
            [[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]],
            [[[0, 1, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]],
        ]
        y_pred = [
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
            [[[0.5, 0.2, 0.3], [0.1, 0.7, 0.2]], [[0.55, 0.25, 0.3], [0.6, 0.1, 0.3]]],
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
        ]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        # target
        target = tf.constant(
            [
                [[0.10258661, 23.025852], [5.725661, 3.218876]],
                [[0.6931472, 0.71334994], [6.4964147, 0.5108256]],
                [[0.10258661, 23.025852], [5.725661, 3.218876]],
            ]
        )

        # calculate
        result = bg_weighted_categorical_crossentropy(y_true, y_pred, 1, 5, 10, 2)

        # assertion
        tf.debugging.assert_equal(target, result)
