import tensorflow as tf
from image_keras.tf.keras.layers.extract_patch_layer import ExtractPatchLayer
from tensorflow.keras.layers import Layer, Reshape


class AggregationLayer2(Layer):
    """
    Aggregation Layer to collect result from neighbor probability to bins.
    """

    def __init__(
        self,
        bin_size: int,
        k_size: int = 5,
        aggregate_mode: str = "weighted_sum",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.k_size = k_size
        self.aggregate_mode = aggregate_mode

    def build(self, input_shape):
        self.input_main_batch_size = input_shape[0][0]
        self.input_main_h_size = input_shape[0][1]
        self.input_main_channels_size = input_shape[0][-1]

    def get_config(self):
        config = super(AggregationLayer2, self).get_config()
        config.update(
            {
                "bin_size": self.bin_size,
                "k_size": self.k_size,
                "aggregate_mode": self.aggregate_mode,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        attn = inputs[0]
        ref_value = inputs[1]
        current_ref_value = inputs[2]

        # Aggregate
        if self.aggregate_mode == "weighted_sum":
            ref_value_stacked = ExtractPatchLayer(k_size=self.k_size)(ref_value)
            ref_value_stacked = Reshape(
                (
                    self.input_main_h_size,
                    self.input_main_h_size,
                    self.k_size * self.k_size,
                    self.bin_size,
                )
            )(ref_value_stacked)
            current_ref_value = tf.expand_dims(current_ref_value, axis=-2)
            ref_value_stacked = tf.concat(
                [ref_value_stacked, current_ref_value], axis=-2
            )
            aggregation = tf.einsum("bhwk,bhwki->bhwi", attn, ref_value_stacked)
        elif self.aggregate_mode == "argmax":
            raise NotImplementedError("`argmax` aggregate has not been implemented yet")
            ref_value_stacked = ExtractPatchLayer(k_size=self.k_size)(ref_value)
            ref_value_stacked = tf.reshape(
                ref_value_stacked,
                (
                    -1,
                    self.h_size,
                    self.h_size,
                    self.k_size * self.k_size,
                    self.bin_size,
                ),
            )
            attn2 = tf.one_hot(
                tf.argmax(attn, axis=-1), depth=tf.shape(attn)[-1], dtype=tf.float32
            )
            one_hot = tf.one_hot(tf.cast(output, tf.int32), num_words, dtype=tf.float32)
            aggregation = tf.einsum("bhwk,bhwki->bhwi", attn2, ref_value_stacked)
        else:
            raise ValueError(
                "`aggregate_mode` value is not valid. Should be one of 'weighted_sum', 'argmax'."
            )
        return aggregation
