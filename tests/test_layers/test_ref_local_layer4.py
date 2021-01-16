import tensorflow as tf
from layers.ref_local_layer4 import RefLocal4


class RefLocalLayer4Test(tf.test.TestCase):
    def test_ref_local_layer_dot(self):
        batch_size = 1
        hw_size = 2
        channel_size = 3
        k_size = 3
        bin_size = 5

        img = tf.reshape(
            tf.range(
                0, (batch_size * hw_size * hw_size * channel_size), dtype=tf.float32
            ),
            (-1, hw_size, hw_size, channel_size),
        )
        main_value = tf.constant(
            [
                [
                    [[0.2, 0.6, 0.2, 0.0, 0.0], [0.7, 0.1, 0.1, 0.0, 0.1]],
                    [[0.35, 0.25, 0.15, 0.0, 0.25], [0.2, 0.6, 0.1, 0.0, 0.1]],
                ]
            ]
        )
        ref = tf.reshape(
            tf.range(
                batch_size * hw_size * hw_size * channel_size,
                (batch_size * hw_size * hw_size * channel_size * 2),
                dtype=tf.float32,
            ),
            (-1, hw_size, hw_size, channel_size),
        )
        ref_value = tf.constant(
            [
                [
                    [[0.5, 0.3, 0.1, 0.0, 0.0], [0.6, 0.0, 0.3, 0.0, 0.1]],
                    [[0.2, 0.15, 0.25, 0.0, 0.3], [0.1, 0.7, 0.0, 0.0, 0.2]],
                ]
            ]
        )

        ref_local = RefLocal4(
            bin_size=bin_size, k_size=k_size, mode="dot", aggregate_mode="weighted_sum"
        )([img, main_value, ref, ref_value])
        print(ref_local)

        # ref_local = RefLocal4(
        #     bin_size=bin_size, k_size=k_size, mode="dot", aggregate_mode="argmax"
        # )([img, main_value, ref, ref_value])
        # print(ref_local)
