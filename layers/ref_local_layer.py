import tensorflow as tf
from tensorflow.keras.layers import Layer

from layers.extract_patch_layer2 import ExtractPatchLayer2


class RefLocal(Layer):
    def __init__(self, bin_size: int, k_size: int = 5, mode: str = "dot", **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.k_size = k_size
        self.mode = mode

    def build(self, input_shape):
        self.custom_shape = input_shape[0]

    def get_config(self):
        config = super(RefLocal, self).get_config()
        config.update(
            {"bin_size": self.bin_size, "k_size": self.k_size, "mode": self.mode}
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        main = inputs[0]
        ref = inputs[1]
        ref_value = inputs[2]

        main = tf.expand_dims(main, axis=-2)

        ref_stacked = ExtractPatchLayer2(k_size=self.k_size)(ref)
        ref_stacked = tf.reshape(
            ref_stacked,
            (
                -1,
                self.custom_shape[-3],
                self.custom_shape[-2],
                self.k_size * self.k_size,
                self.custom_shape[-1],
            ),
        )

        ref_value_stacked = ExtractPatchLayer2(k_size=self.k_size)(ref_value)
        ref_value_stacked = tf.reshape(
            ref_value_stacked,
            (
                -1,
                self.custom_shape[-3],
                self.custom_shape[-2],
                self.k_size * self.k_size,
                self.bin_size,
            ),
        )

        attn = None
        if self.mode == "dot":
            attn = ref_stacked * main
            attn = tf.reduce_sum(attn, axis=4)
            # attn = tf.reduce_sum(attn, axis=-1)
            attn = tf.nn.softmax(attn, axis=3)
            # attn = tf.nn.softmax(attn, axis=-2)
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

        attn = tf.expand_dims(attn, axis=-1)
        stacked_multiply = attn * ref_value_stacked
        stacked_sum = tf.reduce_sum(stacked_multiply, axis=3)
        # stacked_sum = tf.reduce_sum(stacked_multiply, axis=-2)
        return stacked_sum
