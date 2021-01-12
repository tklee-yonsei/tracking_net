import tensorflow as tf
from layers.extract_patch_layer import ExtractPatchLayer


class ExtractPatchLayerTest(tf.test.TestCase):
    def test_extract_patch_layer_shapes(self):
        batch_size = 2
        hw_size = 2
        channel_size = 3
        k_size = 3

        input_feature = tf.ones([batch_size, hw_size, hw_size, channel_size])
        output = ExtractPatchLayer(k_size=k_size)(input_feature)

        # assertion
        tf.debugging.assert_shapes(
            [(output, (batch_size, hw_size, hw_size, k_size * k_size * channel_size))]
        )
