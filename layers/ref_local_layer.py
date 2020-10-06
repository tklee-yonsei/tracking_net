import tensorflow as tf
from keras.layers import Layer


class RefLocal(Layer):
    def __init__(self, bin_size: int, k_size: int = 5, mode: str = "dot", **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.k_size = k_size
        self.mode = mode

    def build(self, input_shape):
        print("RefLocal input_shape: {}".format(input_shape))
        self.custom_input_shape = input_shape
        self.custom_shape = input_shape[0]
        self.batch_size = self.custom_shape[0]
        self.channels_size = self.custom_shape[-1]

    def call(self, inputs):
        self.main = inputs[0]
        self.ref = inputs[1]
        self.ref_value = inputs[2]

        main = tf.reshape(
            self.main,
            [-1, self.custom_shape[1], self.custom_shape[2], 1, self.custom_shape[3]],
        )

        ref_reshaped = tf.reshape(
            self.ref,
            [-1, 1, self.custom_shape[1], self.custom_shape[2], self.custom_shape[3]],
        )
        ref_stacked = tf.extract_volume_patches(
            input=ref_reshaped,
            ksizes=[1, 1, self.k_size, self.k_size, 1],
            strides=[1, 1, 1, 1, 1],
            padding="SAME",
        )
        ref_stacked = tf.reshape(
            ref_stacked,
            (
                -1,
                self.custom_shape[1],
                self.custom_shape[2],
                self.k_size * self.k_size,
                self.custom_shape[3],
            ),
        )

        ref_value_reshaped = tf.reshape(
            self.ref_value,
            [-1, 1, self.custom_shape[1], self.custom_shape[2], self.bin_size],
        )
        ref_value_stacked = tf.extract_volume_patches(
            input=ref_value_reshaped,
            ksizes=[1, 1, self.k_size, self.k_size, 1],
            strides=[1, 1, 1, 1, 1],
            padding="SAME",
        )
        ref_value_stacked = tf.reshape(
            ref_value_stacked,
            (
                -1,
                self.custom_shape[1],
                self.custom_shape[2],
                self.k_size * self.k_size,
                self.bin_size,
            ),
        )

        attn = None
        if self.mode == "dot":
            attn = ref_stacked * main
            attn = tf.reduce_sum(attn, axis=4)
            attn = tf.nn.softmax(attn, axis=3)
            print("dot attn.shape: {}".format(attn.shape))
        elif self.mode == "norm_dot":
            ref_con = tf.concat([ref_stacked, main], -2)
            attn = ref_con * main
            attn = tf.reduce_sum(attn, axis=4)
            attn = tf.nn.softmax(attn, axis=3)
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
            attn = tf.reduce_sum(attn, axis=4)
            attn = tf.nn.softmax(attn, axis=3)
            print("else attn.shape: {}".format(attn.shape))

        attn = tf.reshape(attn, (-1, attn.shape[1], attn.shape[2], attn.shape[3], 1))
        print("attn.shape: {}".format(attn.shape))
        stacked_multiply = attn * ref_value_stacked
        print("stacked_multiply.shape: {}".format(stacked_multiply.shape))
        stacked_sum = tf.reduce_sum(stacked_multiply, axis=3)
        print("stacked_sum.shape: {}".format(stacked_sum.shape))

        return stacked_sum
