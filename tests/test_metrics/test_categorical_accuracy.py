from unittest import TestCase

import tensorflow as tf
from metrics.categorical_accuracy import CategoricalAccuracy, categorical_accuracy


class TestCategoricalAccuracy(TestCase):
    def test_categorical_accuracy(self):
        y_true = [[0, 0, 1], [0, 1, 0]]
        y_pred = [[0.05, 0.9, 0.05], [0.05, 0.95, 0]]

        # target
        ground_truth = tf.constant([0.0, 1.0])

        # calculate
        result = categorical_accuracy(y_true, y_pred)

        # assertion
        tf.debugging.assert_equal(ground_truth, result)

    def test_categorical_accuracy_with_class(self):
        y_true = [[0, 0, 1], [0, 1, 0]]
        y_pred = [[0.05, 0.9, 0.8], [0.05, 0.95, 0]]
        sample_weight = [0.7, 0.3]

        # target
        ground_truth1 = tf.constant(0.5)
        ground_truth2 = tf.constant(0.3)

        # calculate
        categorical_accuracy = CategoricalAccuracy()
        categorical_accuracy.update_state(y_true, y_pred)
        result1 = categorical_accuracy.result()

        categorical_accuracy.reset_states()
        categorical_accuracy.update_state(y_true, y_pred, sample_weight=sample_weight)
        result2 = categorical_accuracy.result()

        # assertion
        tf.debugging.assert_equal(ground_truth1, result1)
        tf.debugging.assert_equal(ground_truth2, result2)
