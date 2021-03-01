from unittest import TestCase

from metrics.cosine_similarity import CosineSimilarity

import tensorflow as tf


class TestCosineSimilarity(TestCase):
    def test_cosine_similarity(self):
        y_true = [[0.0, 1.0], [1.0, 1.0]]
        y_pred = [[1.0, 0.0], [1.0, 1.0]]

        # target
        l2_norm_y_true = [
            [0.0 / (0.0 ** 2 + 1.0 ** 2) ** 0.5, 1.0 / (0.0 ** 2 + 1.0 ** 2) ** 0.5],
            [1.0 / (1.0 ** 2 + 1.0 ** 2) ** 0.5, 1.0 / (1.0 ** 2 + 1.0 ** 2) ** 0.5],
        ]
        l2_norm_y_pred = [
            [1.0 / (1.0 ** 2 + 0.0 ** 2) ** 0.5, 0.0 / (1.0 ** 2 + 0.0 ** 2) ** 0.5],
            [1.0 / (1.0 ** 2 + 1.0 ** 2) ** 0.5, 1.0 / (1.0 ** 2 + 1.0 ** 2) ** 0.5],
        ]
        y_true_pred_dot = tf.constant(l2_norm_y_true) * tf.constant(l2_norm_y_pred)
        y_true_pred_dot_sum = tf.reduce_sum(y_true_pred_dot, axis=1)
        target_cosine_simility_y = tf.reduce_mean(y_true_pred_dot_sum)

        # calculate
        cosine_simility = CosineSimilarity(axis=1)
        cosine_simility.update_state(y_true, y_pred)
        result = cosine_simility.result()

        # assertion
        tf.debugging.assert_equal(target_cosine_simility_y, result)

    def test_cosine_similarity2(self):
        y_true = [[0.0, 1.0], [1.0, 1.0]]
        y_pred = [[1.0, 0.0], [1.0, 1.0]]
        sample_weight = [0.3, 0.7]

        # target
        l2_norm_y_true = [
            [0.0 / (0.0 ** 2 + 1.0 ** 2) ** 0.5, 1.0 / (0.0 ** 2 + 1.0 ** 2) ** 0.5],
            [1.0 / (1.0 ** 2 + 1.0 ** 2) ** 0.5, 1.0 / (1.0 ** 2 + 1.0 ** 2) ** 0.5],
        ]
        l2_norm_y_pred = [
            [1.0 / (1.0 ** 2 + 0.0 ** 2) ** 0.5, 0.0 / (1.0 ** 2 + 0.0 ** 2) ** 0.5],
            [1.0 / (1.0 ** 2 + 1.0 ** 2) ** 0.5, 1.0 / (1.0 ** 2 + 1.0 ** 2) ** 0.5],
        ]
        y_true_pred_dot = tf.constant(l2_norm_y_true) * tf.constant(l2_norm_y_pred)
        y_true_pred_dot_sum = tf.reduce_sum(y_true_pred_dot, axis=1)
        target_cosine_simility_y = tf.reduce_mean(y_true_pred_dot_sum)
        target_cosine_simility_y = 0.6999999  # How to weight for `sample_weight`?

        # calculate (with reset)
        cosine_simility = CosineSimilarity(axis=1)
        cosine_simility.update_state(y_true, y_pred)
        cosine_simility.reset_states()
        cosine_simility.update_state(y_true, y_pred, sample_weight)
        result = cosine_simility.result()

        # assertion
        tf.debugging.assert_equal(target_cosine_simility_y, result)
