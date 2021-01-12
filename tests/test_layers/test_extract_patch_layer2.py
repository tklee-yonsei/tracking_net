import tensorflow as tf
from layers.extract_patch_layer import ExtractPatchLayer
from layers.extract_patch_layer2 import ExtractPatchLayer2


class ExtractPatchLayer2Test(tf.test.TestCase):
    def test_extract_patch_layer2_shapes(self):
        batch_size = 2
        hw_size = 2
        channel_size = 3
        k_size = 3

        input_feature = tf.ones([batch_size, hw_size, hw_size, channel_size])
        output = ExtractPatchLayer2(k_size=k_size)(input_feature)
        output_patch_layer_1 = ExtractPatchLayer(k_size=k_size)(input_feature)

        # assertion
        tf.debugging.assert_shapes(
            [(output, (batch_size, hw_size, hw_size, k_size * k_size * channel_size))]
        )
        tf.debugging.assert_equal(output, output_patch_layer_1)

    def test_extract_patch_layer_shapes_error_even_kernel_size(self):
        batch_size = 2
        hw_size = 2
        channel_size = 3
        k_size = 2

        input_feature = tf.ones([batch_size, hw_size, hw_size, channel_size])

        # assertion
        with self.assertRaises(RuntimeError):
            ExtractPatchLayer2(k_size=k_size)(input_feature)
