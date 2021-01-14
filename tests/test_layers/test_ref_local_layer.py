import numpy as np
import tensorflow as tf
from keras import backend as K
from layers.extract_patch_layer import ExtractPatchLayer
from layers.ref_local_layer import RefLocal


class RefLocalLayerTest(tf.test.TestCase):
    def test_ref_local_layer_dot(self):
        batch_size = 2
        hw_size = 2
        channel_size = 3
        k_size = 3
        bin_size = 4

        input_main = tf.ones([batch_size, hw_size, hw_size, channel_size])
        input_ref = tf.ones([batch_size, hw_size, hw_size, channel_size])
        input_ref_value = tf.ones([batch_size, hw_size, hw_size, bin_size])

        l = RefLocal(bin_size=bin_size, k_size=k_size, mode="dot")(
            [input_main, input_ref, input_ref_value]
        )

    def test_extract_dot_einsum(self):
        batch_size = 1
        hw_size = 2
        channel_size = 3
        k_size = 3

        img = tf.reshape(
            tf.range(
                0, (batch_size * hw_size * hw_size * channel_size), dtype=tf.float32
            ),
            (-1, hw_size, hw_size, channel_size),
        )
        img_result = tf.constant(
            [[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=tf.float32
        )
        # tf.Tensor(
        # [[[[ 0.  1.  2.]
        # [ 3.  4.  5.]]

        # [[ 6.  7.  8.]
        # [ 9. 10. 11.]]]], shape=(1, 2, 2, 3), dtype=float32)
        tf.debugging.assert_equal(img, img_result)

        ref = tf.reshape(
            tf.range(
                batch_size * hw_size * hw_size * channel_size,
                (batch_size * hw_size * hw_size * channel_size * 2),
                dtype=tf.float32,
            ),
            (-1, hw_size, hw_size, channel_size),
        )
        ref_result = tf.constant(
            [[[[12, 13, 14], [15, 16, 17]], [[18, 19, 20], [21, 22, 23]]]],
            dtype=tf.float32,
        )
        # tf.Tensor(
        # [[[[12. 13. 14.]
        # [15. 16. 17.]]

        # [[18. 19. 20.]
        # [21. 22. 23.]]]], shape=(1, 2, 2, 3), dtype=float32)
        tf.debugging.assert_equal(ref, ref_result)

        ref = ExtractPatchLayer(k_size=k_size)(ref)
        # tf.Tensor(
        # [[[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. 12. 13. 14. 15. 16.
        #     17.  0.  0.  0. 18. 19. 20. 21. 22. 23.]
        # [ 0.  0.  0.  0.  0.  0.  0.  0.  0. 12. 13. 14. 15. 16. 17.  0.  0.
        #     0. 18. 19. 20. 21. 22. 23.  0.  0.  0.]]

        # [[ 0.  0.  0. 12. 13. 14. 15. 16. 17.  0.  0.  0. 18. 19. 20. 21. 22.
        #     23.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        # [12. 13. 14. 15. 16. 17.  0.  0.  0. 18. 19. 20. 21. 22. 23.  0.  0.
        #     0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]]]], shape=(1, 2, 2, 27), dtype=float32)

        ref = tf.reshape(ref, (-1, hw_size, hw_size, k_size * k_size, channel_size))
        # tf.Tensor(
        # [[[[[ 0.  0.  0.]
        #     [ 0.  0.  0.]
        #     [ 0.  0.  0.]
        #     [ 0.  0.  0.]
        #     [12. 13. 14.]
        #     [15. 16. 17.]
        #     [ 0.  0.  0.]
        #     [18. 19. 20.]
        #     [21. 22. 23.]]

        # [[ 0.  0.  0.]
        #     [ 0.  0.  0.]
        #     [ 0.  0.  0.]
        #     [12. 13. 14.]
        #     [15. 16. 17.]
        #     [ 0.  0.  0.]
        #     [18. 19. 20.]
        #     [21. 22. 23.]
        #     [ 0.  0.  0.]]]

        # [[[ 0.  0.  0.]
        #     [12. 13. 14.]
        #     [15. 16. 17.]
        #     [ 0.  0.  0.]
        #     [18. 19. 20.]
        #     [21. 22. 23.]
        #     [ 0.  0.  0.]
        #     [ 0.  0.  0.]
        #     [ 0.  0.  0.]]

        # [[12. 13. 14.]
        #     [15. 16. 17.]
        #     [ 0.  0.  0.]
        #     [18. 19. 20.]
        #     [21. 22. 23.]
        #     [ 0.  0.  0.]
        #     [ 0.  0.  0.]
        #     [ 0.  0.  0.]
        #     [ 0.  0.  0.]]]]], shape=(1, 2, 2, 9, 3), dtype=float32)

        attn = tf.einsum("bhwc,bhwkc->bhwk", img, ref)
        # tf.Tensor(
        # [[[[  0.   0.   0.   0.  41.  50.   0.  59.  68.]
        # [  0.   0.   0. 158. 194.   0. 230. 266.   0.]]

        # [[  0. 275. 338.   0. 401. 464.   0.   0.   0.]
        # [392. 482.   0. 572. 662.   0.   0.   0.   0.]]]], shape=(1, 2, 2, 9), dtype=float32)
        attn_ratio = tf.nn.softmax(attn, axis=-1)
        # tf.Tensor(
        # [[[[2.9371197e-30 2.9371197e-30 2.9371197e-30 2.9371197e-30
        #     1.8792971e-12 1.5228100e-08 2.9371197e-30 1.2339458e-04
        #     9.9987662e-01]
        # [0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
        #     5.3801859e-32 0.0000000e+00 2.3195230e-16 1.0000000e+00
        #     0.0000000e+00]]

        # [[0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
        #     4.3596101e-28 1.0000000e+00 0.0000000e+00 0.0000000e+00
        #     0.0000000e+00]
        # [0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
        #     1.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
        #     0.0000000e+00]]]], shape=(1, 2, 2, 9), dtype=float32)

    def test_dot(self):
        batch_size = 1
        hw_size = 2
        channel_size = 3
        k_size = 3

        img = tf.reshape(
            tf.range(
                0, (batch_size * hw_size * hw_size * channel_size), dtype=tf.float32
            )
            / 10,
            (-1, hw_size, hw_size, channel_size),
        )
        img2 = tf.expand_dims(img, -2)

        ref = tf.reshape(
            tf.range(
                batch_size * hw_size * hw_size * channel_size,
                (batch_size * hw_size * hw_size * channel_size * 2),
                dtype=tf.float32,
            )
            / 10,
            (-1, hw_size, hw_size, channel_size),
        )
        ref = ExtractPatchLayer(k_size=k_size)(ref)
        ref = tf.reshape(ref, (-1, hw_size, hw_size, k_size * k_size, channel_size))

        attn_einsum = tf.einsum("bhwc,bhwkc->bhwk", img, ref)
        attn = tf.reduce_sum(ref * img2, axis=-1)
        attn2 = K.squeeze(tf.matmul(img2, ref, transpose_b=True), axis=-2)

        # assertion
        tf.debugging.assert_equal(attn_einsum, attn)
        tf.debugging.assert_equal(attn_einsum, attn2)

        print(attn_einsum)
        attn_ratio = tf.nn.softmax(attn_einsum, axis=-1)
        print(attn_ratio)

    def test_dot_sample(self):
        k_size = 3

        img = tf.constant(
            np.load("tests/test_resources/sample_feature_map/015_02_15_l4_zero.npy")
        )
        img = tf.expand_dims(img, 0)
        hw_size = tf.shape(img)[1]
        channel_size = tf.shape(img)[-1]

        ref = tf.constant(
            np.load("tests/test_resources/sample_feature_map/015_02_15_l4_p1.npy")
        )
        ref = tf.expand_dims(ref, 0)
        ref = ExtractPatchLayer(k_size=k_size)(ref)
        ref = tf.reshape(ref, (-1, hw_size, hw_size, k_size * k_size, channel_size))

        attn_einsum = tf.einsum("bhwc,bhwkc->bhwk", img, ref)

        print(attn_einsum)
        attn_ratio = tf.nn.softmax(attn_einsum, axis=-1)
        for i in range(tf.shape(attn_ratio)[1]):
            for j in range(tf.shape(attn_ratio)[2]):
                print("{}/{}".format(i, j))
                print(attn_ratio[:, i : i + 1, j : j + 1, :])
                print("--------------")
        # total - 9216
        # 0.0 - 2011
        # 0.1 - 7119
        # 0.2 - 71
        # 0.3 - 9
        # 0.4 - 3
        # 0.5 - 3

        print(attn_ratio)
        # np.save("015_02_15_l4", attn_ratio)
        print(attn_ratio[:, 31, 31, :])
