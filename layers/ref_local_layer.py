from functools import reduce
from typing import Generator, List, Optional

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Conv2D, Lambda, Layer, MaxPool1D, Reshape, add, dot

# def affinity(func):
#     def wrapper(image, ref, size):
#         assert image.shape == ref.shape
#         H, W, F = image.shape
#         K = size
#         image = tf.reshape(image, [H, W, 1, F])
#         ref_stacked = tf.image.extract_patches(
#             images=tf.reshape(ref, [1, H, W, F]),
#             sizes=[1, K, K, 1],
#             strides=[1, 1, 1, 1],
#             rates=[1, 1, 1, 1],
#             padding="SAME",
#         )
#         ref_stacked = tf.reshape(ref_stacked, (H, W, K * K, F))
#         return func(image, ref_stacked, size)

#     return wrapper


# def dotprod_util(image, ref):
#     attn = ref * image
#     attn = tf.reduce_sum(attn, axis=3)
#     attn = tf.nn.softmax(attn, axis=2)
#     return attn


# @affinity
# def affinity_dotprod(image, ref_stacked, size):
#     return dotprod_util(image, ref_stacked)


# @affinity
# def affinity_norm_dotprod(image, ref_stacked, size):
#     H, W, _, F = image.shape
#     image_reshape = tf.reshape(image, (H, W, 1, F))
#     ref_con = tf.concat([ref_stacked, image_reshape], -2)
#     attn = dotprod_util(image, ref_con)
#     attn = attn[:, :, :-1]
#     return attn


# @affinity
# def affinity_dotprod_nosoft(image, ref_stacked, size):
#     attn = ref_stacked * image
#     attn = tf.reduce_sum(attn, axis=3)
#     return attn


# @affinity
# def affinity_gaussian(image, ref_stacked, size):
#     attn = tf.math.exp(
#         -tf.reduce_sum(tf.math.squared_difference(image, ref_stacked), axis=-1)
#     )
#     return attn


class RefLocal(Layer):
    def __init__(self, mode: str = "dot", **kwargs):
        super().__init__(**kwargs)
        self.mode = mode

    def build(self, input_shape, k_size):
        self.input_shape = input_shape
        self.k_size = k_size
        self.batch_size = input_shape[0]
        self.channels_size = self.input_shape[-1]
        super(RefLocal, self).build(input_shape)

    def call(self, inputs):
        self.main = inputs[0]
        self.ref = inputs[1]
        self.ref_value = inputs[2]

        main = tf.reshape(
            self.main,
            [self.input_shape[0], self.input_shape[1], 1, self.input_shape[2]],
        )
        ref_stacked = tf.image.extract_patches(
            images=tf.reshape(
                self.ref,
                [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]],
            ),
            sizes=[1, self.k_size, self.k_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        ref_value_stacked = tf.image.extract_patches(
            images=tf.reshape(
                self.ref_value,
                [1, self.input_shape[0], self.input_shape[1], self.input_shape[2]],
            ),
            sizes=[1, self.k_size, self.k_size, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        attn = None
        if self.mode == "dot":
            attn = ref_stacked * main
            attn = tf.reduce_sum(attn, axis=3)
            attn = tf.nn.softmax(attn, axis=2)
            print("dot attn.shape: {}".format(attn.shape))
        elif self.mode == "norm_dot":
            ref_con = tf.concat([ref_stacked, main], -2)
            attn = ref_con * main
            attn = tf.reduce_sum(attn, axis=3)
            attn = tf.nn.softmax(attn, axis=2)
            attn = attn[:, :, :-1]
            print("norm_dot attn.shape: {}".format(attn.shape))
            raise NotImplementedError("norm_dot model has not been implemented yet")
        elif self.mode == "gaussian":
            attn = tf.math.exp(
                -tf.reduce_sum(tf.math.squared_difference(main, ref_stacked), axis=-1)
            )
            print("gaussian attn.shape: {}".format(attn.shape))
            raise NotImplementedError("gaussian model has not been implemented yet")
        else:
            # dot
            attn = ref_stacked * main
            attn = tf.reduce_sum(attn, axis=3)
            attn = tf.nn.softmax(attn, axis=2)
            print("dot attn.shape: {}".format(attn.shape))

        stacked_multiply = attn * ref_value_stacked
        stacked_sum = tf.reduce_sum(stacked_multiply, axis=(2, 3))

        return stacked_sum

    #     batch_size = K.shape(self.main)[0]

    #     # 1) f path
    #     # Gaussian
    #     if self.mode == "gaussian":
    #         x1 = Reshape((-1, self.channels_size))(inputs)  # xi
    #         x2 = Reshape((-1, self.channels_size))(inputs)  # xj
    #         f = dot([x1, x2], axes=2)
    #         f = Activation("softmax")(f)

    #     # Dot
    #     elif self.mode == "dot":
    #         # theta path
    #         theta = Conv2D(
    #             self.channels_size,
    #             (1, 1),
    #             padding="same",
    #             use_bias=False,
    #             kernel_initializer="he_normal",
    #         )(inputs)
    #         theta = Reshape((-1, self.intermediate_dim))(theta)

    #         # phi path
    #         phi = Conv2D(
    #             self.channels_size,
    #             (1, 1),
    #             padding="same",
    #             use_bias=False,
    #             kernel_initializer="he_normal",
    #         )(inputs)
    #         phi = Reshape((-1, self.intermediate_dim))(phi)

    #         f = dot([theta, phi], axes=2)

    #         size = K.int_shape(f)

    #         f = Lambda(lambda z: (1.0 / float(size[-1])) * z)(f)

    #     # Concatenate
    #     elif self.mode == "concatenate":
    #         raise NotImplementedError("concatenate model has not been implemented yet")

    #     # Embedded Gaussian instantiation
    #     else:
    #         # theta path
    #         theta = Conv2D(
    #             self.channels_size,
    #             (1, 1),
    #             padding="same",
    #             use_bias=False,
    #             kernel_initializer="he_normal",
    #         )(inputs)
    #         theta = Reshape((-1, self.intermediate_dim))(theta)

    #         # phi path
    #         phi = Conv2D(
    #             self.channels_size,
    #             (1, 1),
    #             padding="same",
    #             use_bias=False,
    #             kernel_initializer="he_normal",
    #         )(inputs)
    #         phi = Reshape((-1, self.intermediate_dim))(phi)

    #         if self.sub_sample:
    #             phi = MaxPool1D()(phi)

    #         f = dot([theta, phi], axes=2)
    #         f = Activation("softmax")(f)

    #     # 2) g path
    #     g = Conv2D(
    #         self.channels_size,
    #         (1, 1),
    #         padding="same",
    #         use_bias=False,
    #         kernel_initializer="he_normal",
    #     )(inputs)
    #     g = Reshape((-1, self.intermediate_dim))(g)

    #     if self.sub_sample and self.mode == "embedded":
    #         g = MaxPool1D()(g)

    #     # 3) compute output path
    #     y = dot([f, g], axes=[2, 1])

    #     # reshape to input tensor format
    #     y = Reshape((self.input_shape[1], self.input_shape[2], self.intermediate_dim))(
    #         y
    #     )

    #     # project filters
    #     theta = Conv2D(
    #         self.channels_size,
    #         (1, 1),
    #         padding="same",
    #         use_bias=False,
    #         kernel_initializer="he_normal",
    #     )(y)

    #     # residual connection
    #     y = add([inputs, y])

    #     return y

    # def compute_output_shape(self, input_shape):
    #     return input_shape


# from functools import reduce
# from typing import Generator, List


# def aa(f=0, t=10):
#     for i in range(f, t):
#         yield i


# ax1 = aa(0, 10)
# ax2 = aa(5, 15)
# ax3 = aa(16, 29)
# ax4 = aa(30, 66)


# ax_list: List[Generator] = [ax1, ax2, ax3, ax4]


# def zip_generators(generator_list: List[Generator]):
#     def _tuple_reducer(a, b):
#         for _element in zip(a, b):
#             if isinstance(_element[0], tuple):
#                 yield _element[0] + (_element[1],)
#             else:
#                 yield (_element[0],) + (_element[1],)

#     return reduce(lambda a, b: _tuple_reducer(a, b), generator_list)


# cc_ax12 = map(list, zip_generators(ax_list))
# next(cc_ax12)
