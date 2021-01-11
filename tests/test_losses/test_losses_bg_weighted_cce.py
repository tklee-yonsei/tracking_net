from unittest import TestCase

import tensorflow as tf
from losses.losses_bg_weighted_cce import bg_weighted_crossentropy, get_bg_weights


class TestBGWeightedCCE(TestCase):
    def test_get_bg_weights(self):
        # data
        y_true = [
            [[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]],
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]],
            [[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]],
        ]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = [
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
            [[[0.5, 0.2, 0.3], [0.1, 0.7, 0.2]], [[0.55, 0.25, 0.3], [0.6, 0.1, 0.3]]],
            [[[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]], [[0.3, 0.35, 0.35], [0.7, 0.3, 0.0]]],
            [[[0.1, 0.9, 0], [0.9, 0.0, 0.1]], [[0.7, 0.3, 0.0], [0.85, 0.15, 0.0]]],
        ]
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        # target
        target = tf.ones(
            (tf.shape(y_true)[0], tf.shape(y_true)[1], tf.shape(y_true)[2], 1)
        )

        # calculate
        weights = get_bg_weights(
            y_pred,
            y_true,
            bg_to_bg_weight=1,
            bg_to_fg_weight=1,
            fg_to_bg_weight=1,
            fg_to_fg_weight=1,
        )

        # assertion
        self.assertTrue(tf.math.reduce_all(tf.math.equal(weights, target)))

    def test_bg_weighted_crossentropy_1(self):
        # data
        y_true = [
            [[[0, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]]],
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]],
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 0, 0]]],
            [[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [1, 0, 0]]],
        ]
        y_true = tf.constant(y_true, dtype=tf.float32)
        y_pred = [
            [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.45, 0.35, 0.3], [0.2, 0.2, 0.6]]],
            [[[0.5, 0.2, 0.3], [0.1, 0.7, 0.2]], [[0.55, 0.25, 0.3], [0.6, 0.1, 0.3]]],
            [[[0.2, 0.6, 0.2], [0.1, 0.8, 0.1]], [[0.3, 0.35, 0.35], [0.7, 0.3, 0.0]]],
            [[[0.1, 0.9, 0], [0.9, 0.0, 0.1]], [[0.7, 0.3, 0.0], [0.85, 0.15, 0.0]]],
        ]
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        # target
        target = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        # calculate
        result = bg_weighted_crossentropy(
            y_true,
            y_pred,
            bg_to_bg_weight=1,
            bg_to_fg_weight=1,
            fg_to_bg_weight=1,
            fg_to_fg_weight=1,
        )

        # assertion
        self.assertTrue(tf.math.reduce_all(tf.math.equal(result, target)))
