import tensorflow as tf
from image_keras.tf.keras.layers.extract_patch_layer import ExtractPatchLayer
from tensorflow.keras.layers import Layer


class RefLocal2(Layer):
    def __init__(self, bin_size: int, k_size: int = 5, mode: str = "dot", **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.k_size = k_size
        self.mode = mode

    def build(self, input_shape):
        self.custom_input_shape = input_shape
        self.custom_shape = input_shape[0]
        self.batch_size = input_shape[0][0]
        self.channels_size = input_shape[0][-1]

    def get_config(self):
        config = super(RefLocal2, self).get_config()
        config.update(
            {"bin_size": self.bin_size, "k_size": self.k_size, "mode": self.mode}
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        main = inputs[0]
        main_value = inputs[1]
        ref = inputs[2]
        ref_value = inputs[3]

        main = tf.expand_dims(main, axis=-2)

        main_value = tf.expand_dims(main_value, axis=-2)

        ref_stacked = ExtractPatchLayer(k_size=self.k_size)(ref)
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

        ref_value_stacked = ExtractPatchLayer(k_size=self.k_size)(ref_value)
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
        stacked_sum = None
        if self.mode == "dot":
            ref_con = tf.concat([ref_stacked, main], -2)
            attn = ref_con * main
            attn = tf.reduce_sum(attn, axis=4)
            attn = tf.nn.softmax(attn, axis=3)
            # attn = attn[:, :, :-1]
            attn = tf.expand_dims(attn, axis=-1)

            ref_value_stacked_con = tf.concat([ref_value_stacked, main_value], -2)
            stacked_multiply = attn * ref_value_stacked_con
            stacked_sum = tf.reduce_sum(stacked_multiply, axis=3)
        elif self.mode == "norm_dot":
            # ref_con = tf.concat([ref_stacked, main], -2)
            # attn = ref_con * main
            # attn = tf.reduce_sum(attn, axis=4)
            # attn = tf.nn.softmax(attn, axis=3)
            # attn = attn[:, :, :-1]
            # print("norm_dot attn.shape: {}".format(attn.shape))
            raise NotImplementedError("norm_dot model has not been implemented yet")
        elif self.mode == "gaussian":
            # attn = tf.math.exp(
            #     -tf.reduce_sum(tf.math.squared_difference(main, ref_stacked), axis=-1)
            # )
            # print("gaussian attn.shape: {}".format(attn.shape))
            raise NotImplementedError("gaussian model has not been implemented yet")
        else:
            # dot
            # attn = ref_stacked * main
            # attn = tf.reduce_sum(attn, axis=4)
            # attn = tf.nn.softmax(attn, axis=3)

            # # attn = tf.reshape(
            # #     attn, (-1, attn.shape[1], attn.shape[2], attn.shape[3], 1)
            # # )
            # attn = tf.reshape(
            #     attn,
            #     (-1, tf.shape(attn)[-3], tf.shape(attn)[-2], tf.shape(attn)[-1], 1),
            # )
            # stacked_multiply = attn * ref_value_stacked
            # stacked_sum = tf.reduce_sum(stacked_multiply, axis=3)
            raise NotImplementedError("gaussian model has not been implemented yet")

        return stacked_sum
