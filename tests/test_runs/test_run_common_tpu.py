import os
from typing import Optional
from unittest import TestCase

from _run.run_common_tpu import (
    check_all_exists_or_not,
    check_both_exists_or_not,
    setup_continuous_training,
)


class TestRUNCommonTPU(TestCase):
    def test_check_both_exists_or_not(self):
        self.assertTrue(check_both_exists_or_not(None, None))
        self.assertTrue(check_both_exists_or_not("exist this", 3))
        self.assertFalse(check_both_exists_or_not("exist this", None))
        self.assertFalse(check_both_exists_or_not(None, 3))

    def test_check_all_exists_or_not(self):
        self.assertTrue(check_all_exists_or_not([None, None]))
        self.assertTrue(check_all_exists_or_not(["exist this", 3]))
        self.assertFalse(check_all_exists_or_not(["exist this", None]))
        self.assertFalse(check_all_exists_or_not([None, 3]))

    def test_setup_continuous_training(self):
        # Expected Result
        continuous_model_name: Optional[str] = None
        continuous_epoch: Optional[int] = 3

        # 1) Get sample image as tf
        setup_continuous_training(continuous_model_name, continuous_epoch)

        # 2) Calculate shape
        # self.assertTrue(tf.math.reduce_all(tf.math.equal(tf.shape(tf_image), result)))

    def test_setup_continuous_training2(self):
        # Expected Result
        # continuous_model_name: Optional[str] = "dfdf."
        continuous_model_name: Optional[
            str
        ] = "gs://cell_dataset/save/weights/training__model_unet_l4__run_leetaekyu_20210108_221742.epoch_78-val_loss_0.179-val_accuracy_0.974"
        continuous_epoch: Optional[int] = 3

        training_id = "3"
        training_id = (
            setup_continuous_training(continuous_model_name, continuous_epoch)
            or training_id
        )
        print(training_id)

        # 2) Calculate shape
        # self.assertTrue(tf.math.reduce_all(tf.math.equal(tf.shape(tf_image), result)))
