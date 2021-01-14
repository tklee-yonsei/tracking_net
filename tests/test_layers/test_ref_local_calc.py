import numpy as np
import tensorflow as tf
from keras import backend as K
from layers.extract_patch_layer import ExtractPatchLayer
from layers.ref_local_layer import RefLocal


class RefLocalCalcTest(tf.test.TestCase):
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

    def test_norm_dot_sample(self):
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

        img2 = tf.expand_dims(img, -2)
        ref_img = tf.concat([ref, img2], axis=-2)

        attn_einsum = tf.einsum("bhwc,bhwkc->bhwk", img, ref_img)
        print(attn_einsum)

        attn_ratio = tf.nn.softmax(attn_einsum, axis=-1)
        attn_ratio = attn_ratio[:, :, :, :-1]
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

    def test_aggregate_mode(self):
        # Attention
        batch_size = 1
        hw_size = 2
        k_size = 3
        attn = tf.reshape(
            tf.range(
                0, (batch_size * hw_size * hw_size * k_size * k_size), dtype=tf.float32
            ),
            (-1, hw_size, hw_size, k_size * k_size),
        )
        # tf.Tensor(
        # [[[[ 0.  1.  2.  3.  4.  5.  6.  7.  8.]
        # [ 9. 10. 11. 12. 13. 14. 15. 16. 17.]]

        # [[18. 19. 20. 21. 22. 23. 24. 25. 26.]
        # [27. 28. 29. 30. 31. 32. 33. 34. 35.]]]], shape=(1, 2, 2, 9), dtype=float32)

        # Aggregate `weighted_sum`
        bin_size = 5
        ref_value = tf.constant(
            [
                [
                    [[0.5, 0.3, 0.1, 0.0, 0.0], [0.6, 0.0, 0.3, 0.0, 0.1]],
                    [[0.2, 0.15, 0.25, 0.0, 0.3], [0.1, 0.7, 0.0, 0.0, 0.2]],
                ]
            ]
        )
        # tf.Tensor(
        # [[[[0.5  0.3  0.1  0.   0.  ]
        # [0.6  0.   0.3  0.   0.1 ]]

        # [[0.2  0.15 0.25 0.   0.3 ]
        # [0.1  0.7  0.   0.   0.2 ]]]], shape=(1, 2, 2, 5), dtype=float32)
        ref_value_stacked = ExtractPatchLayer(k_size=k_size)(ref_value)
        ref_value_stacked = tf.reshape(
            ref_value_stacked, (-1, hw_size, hw_size, k_size * k_size, bin_size),
        )
        # tf.Tensor(
        # [[[[[0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.5  0.3  0.1  0.   0.  ]
        #     [0.6  0.   0.3  0.   0.1 ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.2  0.15 0.25 0.   0.3 ]
        #     [0.1  0.7  0.   0.   0.2 ]]

        # [[0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.5  0.3  0.1  0.   0.  ]
        #     [0.6  0.   0.3  0.   0.1 ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.2  0.15 0.25 0.   0.3 ]
        #     [0.1  0.7  0.   0.   0.2 ]
        #     [0.   0.   0.   0.   0.  ]]]

        # [[[0.   0.   0.   0.   0.  ]
        #     [0.5  0.3  0.1  0.   0.  ]
        #     [0.6  0.   0.3  0.   0.1 ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.2  0.15 0.25 0.   0.3 ]
        #     [0.1  0.7  0.   0.   0.2 ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]]

        # [[0.5  0.3  0.1  0.   0.  ]
        #     [0.6  0.   0.3  0.   0.1 ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.2  0.15 0.25 0.   0.3 ]
        #     [0.1  0.7  0.   0.   0.2 ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]
        #     [0.   0.   0.   0.   0.  ]]]]], shape=(1, 2, 2, 9, 5), dtype=float32)
        aggregation = tf.einsum("bhwk,bhwki->bhwi", attn, ref_value_stacked)
        # tf.Tensor(
        # [[[[ 7.2000003  7.85       3.65       0.         4.2000003]
        # [18.4       17.05       8.85       0.         9.       ]]

        # [[28.199999  25.1       13.4        0.        13.200001 ]
        # [39.4       34.3       18.6        0.        18.       ]]]], shape=(1, 2, 2, 5), dtype=float32)

        # [[[[4*0.5+5*0.6+7*0.2+8*0.1   4*0.3+7*0.15+8*0.7   4*0.1+5*0.3+7*0.25    0    5*0.1+7*0.3+8*0.2]
        # [...]]]]
        attn2 = tf.expand_dims(attn, axis=-1)
        stacked_multiply = attn2 * ref_value_stacked
        aggregation2 = tf.reduce_sum(stacked_multiply, axis=3)

        tf.debugging.assert_equal(aggregation, aggregation2)

    def test_aggregate_mode_argmax(self):
        # Attention
        batch_size = 1
        hw_size = 2
        k_size = 3
        attn = tf.constant(
            [
                [
                    [
                        [0.0, 1.0, 9.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 22.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                    ],
                    [
                        [18.0, 19.0, 29.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
                        [27.0, 28.0, 29.0, 30.0, 31.0, 44.0, 33.0, 34.0, 35.0],
                    ],
                ]
            ]
        )

        # Aggregate `argmax`
        bin_size = 5
        ref_value = tf.constant(
            [
                [
                    [[0.5, 0.3, 0.1, 0.0, 0.0], [0.6, 0.0, 0.3, 0.0, 0.1]],
                    [[0.2, 0.15, 0.25, 0.0, 0.3], [0.1, 0.7, 0.0, 0.0, 0.2]],
                ]
            ]
        )
        ref_value_stacked = ExtractPatchLayer(k_size=k_size)(ref_value)
        ref_value_stacked = tf.reshape(
            ref_value_stacked, (-1, hw_size, hw_size, k_size * k_size, bin_size),
        )
        aggregation = tf.einsum(
            "bhwk,bhwki->bhwi",
            tf.one_hot(tf.argmax(attn, axis=-1), depth=tf.shape(attn)[-1]),
            ref_value_stacked,
        )
        result = tf.constant(
            [
                [
                    [[0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.3, 0.1, 0.0, 0.0]],
                    [[0.6, 0.0, 0.3, 0.0, 0.1], [0.0, 0.0, 0.0, 0.0, 0.0]],
                ]
            ]
        )

        tf.debugging.assert_equal(aggregation, result)
