import numpy as np
import tensorflow as tf


class DenseLayerTest(tf.test.TestCase):
    def setUp(self):
        super(DenseLayerTest, self).setUp()
        self.my_dense = tf.keras.layers.Dense(units=2)
        self.my_dense.build(input_shape=(2, 2))

    def testDenseLayerOutput(self):
        self.my_dense.set_weights([tf.constant([[1, 0], [2, 3]]), np.array([0.5, 0])])
        input_x = tf.constant([[1, 2], [2, 3]])
        output = self.my_dense(input_x)

        expected_output = tf.constant([[5.5, 6.0], [8.5, 9]])

        # assertion
        tf.debugging.assert_equal(expected_output, output)
