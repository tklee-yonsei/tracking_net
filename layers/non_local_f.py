import tensorflow as tf


def affinity(func):
    """
    

    Parameters
    ----------
    func : [type]
        처리할 함수
    """

    def wrapper(image, ref, size):
        assert image.shape == ref.shape
        H, W, F = image.shape
        K = size
        image = tf.reshape(image, (H, W, 1, F))
        ref_stacked = tf.image.extract_patches(
            images=tf.reshape(ref, [1, H, W, F]),
            sizes=[1, K, K, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        ref_stacked = tf.reshape(ref_stacked, (H, W, K * K, F))
        return func(image, ref_stacked, size)

    return wrapper


def dotprod_util(image, ref):
    # attn = ref_stacked * image
    attn = ref * image
    attn = tf.reduce_sum(attn, axis=3)
    attn = tf.nn.softmax(attn, axis=2)
    return attn


@affinity
def affinity_dotprod(image, ref_stacked, size):
    return dotprod_util(image, ref_stacked)


@affinity
def affinity_norm_dotprod(image, ref_stacked, size):
    H, W, F = image.shape
    image_reshape = tf.reshape(image, (H, W, 1, F))
    ref_con = tf.concat([ref_stacked, image_reshape], -2)
    # attn = dotprod_util(image, ref_stacked)
    attn = dotprod_util(image, ref_con)
    attn = attn[:, :, :-1, :]
    return attn


@affinity
def affinity_dotprod_nosoft(image, ref_stacked, size):
    attn = ref_stacked * image
    attn = tf.reduce_sum(attn, axis=3)
    return attn


@affinity
def affinity_gaussian(image, ref_stacked, size):
    attn = tf.math.exp(
        -tf.reduce_sum(tf.math.squared_difference(image, ref_stacked), axis=-1)
    )
    return attn
