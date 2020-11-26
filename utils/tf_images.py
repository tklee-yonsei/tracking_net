from typing import Tuple

import tensorflow as tf
from tensorflow.python.ops import gen_array_ops


def decode_png(filename: str, channels: int = 1):
    """
    Read image from `filename` with `channels`.

    Parameters
    ----------
    filename : str
        A filename to read.
    channels : int, optional
        Number of channel. 3 for RGB, 1 for Grayscale, by default 1

    Returns
    -------
    tf.float32 `Tensor` of Image
        Image Tensor

    Examples
    --------
    >>> import tensorflow as tf
    >>> from utils import tf_images 
    >>> sample_img = tf_images.decode_png("tests/test_resources/sample.png", 3)
    >>> tf.shape(sample_img)
    tf.Tensor([180 180   3], shape=(3,), dtype=int32)
    """
    bits = tf.io.read_file(filename)
    image = tf.image.decode_png(bits, channels)
    image = tf.cast(image, tf.float32)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False)
    return image


def save_img(tf_img, filename: str):
    """
    Save `Tensor` of `tf_img` to file.

    Parameters
    ----------
    tf_img : `Tensor`
        `Tensor` of image.
    filename : str
        File path to save.
    
    Examples
    --------
    >>> from utils import tf_images 
    >>> tf_img = ...
    >>> tf_images.save_img(tf_img, "file_name.png")
    """
    tf.keras.preprocessing.image.save_img(filename, tf_img)


def tf_img_to_minmax(
    tf_img, threshold: float, min_max: Tuple[float, float] = (0.0, 1.0),
):
    """
    Convert grayscale `Tensor` of image, to `Tensor` of `min_max` value.

    Parameters
    ----------
    tf_img : `Tensor` of Image
        Should be grayscale image. (channel 1)
    threshold : float
        Threshold value to determine `min_max`
    min_max : Tuple[float, float], optional
        Min max value for `tf_img`, by default (0.0, 1.0)

    Returns
    -------
    `Tensor` of Image
        In image, exist only min_max values.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from utils import tf_images 
    >>> from tensorflow.python.ops import gen_array_ops
    >>> grayscale_sample_img = tf_images.decode_png("tests/test_resources/sample.png", 1)
    >>> min_maxed_grayscale_tf_image = tf_images.tf_img_to_minmax(
    ...     grayscale_tf_image, 127, (0, 255)
    ... )
    >>> reshaped_min_maxed_grayscale_tf_image = tf.reshape(
    ...     min_maxed_grayscale_tf_image, (-1, 1)
    ... )
    >>> print(gen_array_ops.unique_v2(reshaped_min_maxed_grayscale_tf_image, axis=[-2]))
    UniqueV2(y=<tf.Tensor: shape=(2, 1), dtype=float32, numpy=
    array([[255.],
        [  0.]], dtype=float32)>, idx=<tf.Tensor: shape=(32400,), dtype=int32, numpy=array([0, 0, 0, ..., 0, 0, 0], dtype=int32)>)
    >>> print(tf.math.count_nonzero(min_maxed_grayscale_tf_image))
    tf.Tensor(31760, shape=(), dtype=int64)
    """
    cond = tf.greater(tf_img, tf.ones_like(tf_img) * threshold)
    mask = tf.where(
        cond, tf.ones_like(tf_img) * min_max[1], tf.ones_like(tf_img) * min_max[0]
    )
    return mask


def tf_equalize_histogram(tf_img):
    """
    Tensorflow Image Histogram Equalization

    https://stackoverflow.com/questions/42835247/how-to-implement-histogram-equalization-for-images-in-tensorflow

    Parameters
    ----------
    tf_img : `Tensor` of image
        Input `Tensor` image `tf_img`

    Returns
    -------
    `Tensor` of image
        Equalized histogram image of input `Tensor` image `tf_img`.
    """
    values_range = tf.constant([0.0, 255.0], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(tf.cast(tf_img, tf.float32), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]
    img_shape = tf.shape(tf_img)

    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(
        tf.cast(cdf - cdf_min, tf.float32) * 255.0 / tf.cast(pix_cnt - 1, tf.float32)
    )
    px_map = tf.cast(px_map, tf.uint8)
    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(tf_img, tf.int32)), 2)
    return eq_hist


def tf_get_all_colors(tf_img):
    """
    Get all colors in `Tensor` image `tf_img`.
    The result always contains [0, 0, 0].

    Parameters
    ----------
    tf_img : `Tensor` of image
        Input `Tensor` image `tf_img`. Should be color image.

    Returns
    -------
    `Tensor` array of colors
        All colors in `Tensor` image.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from utils import tf_images 
    >>> sample_img = tf_images.decode_png("tests/test_resources/sample.png", 3)
    >>> print(tf_images.tf_get_all_colors(sample_img))
    tf.Tensor(
    [[245. 245. 245.]
     [ 71.  71.  71.]
     [  0.   0.   0.]
     [255. 145.  77.]
     [ 72.  72.  72.]], shape=(5, 3), dtype=float32)
    """
    scs = tf.reshape(tf_img, (-1, 3))
    scs = gen_array_ops.unique_v2(scs, axis=[-2])[0]
    scs = tf.cond(
        tf.reduce_any(tf.reduce_all(tf.equal(scs, [0, 0, 0]), axis=-1)),
        lambda: scs,
        lambda: tf.concat([tf.constant([[0, 0, 0]], dtype=tf_img.dtype), scs], axis=-2),
    )
    scs = tf.cast(scs, tf.float32)
    return scs


def tf_generate_color_map(img):
    img_color = tf_get_all_colors(img)
    img_color_index = tf.range(tf.shape(img_color)[-2], dtype=tf.float32)
    return img_color_index, img_color


def tf_shrink3D(data, rows: int, cols: int, channels: int):
    """
    Shrink 3D `Tensor` data.

    Parameters
    ----------
    data : `Tensor`
        `Tensor` data to shrink. Shape should be 3-dimension.
    rows : int
        Number of rows
    cols : int
        Number of columns
    channels : int
        Number of channels

    Returns
    -------
    Shrinked `Tensor` array
        Shrinked 3d `Tensor`

    Examples
    --------
    >>> import tensorflow as tf
    >>> from utils import tf_images 
    >>> a = tf.constant(
    ...     np.array(
    ...         [
    ...             [[1, 2], [3, 4]],
    ...             [[5, 6], [7, 8]],
    ...             [[9, 10], [11, 12]],
    ...             [[13, 14], [15, 16]],
    ...         ]
    ...     )
    ... )
    >>> print(shrink3D(a,2,1,2))
    tf.Tensor(
    [[[16 20]]  # [[[   1+3+5+7, 2+4+6+8]],
    [[48 52]]], shape=(2, 1, 2), dtype=int64)   #  [[9+11+13+15, 10+12+14+16]]]
    >>> print(shrink3D(a,2,1,1))
    tf.Tensor(
    [[[ 36]]    # [[[   1+3+5+7+2+4+6+8]],
     [[100]]], shape=(2, 1, 1), dtype=int64)    #  [[9+11+13+15+10+12+14+16]]]
    """
    return tf.reduce_sum(
        tf.reduce_sum(
            tf.reduce_sum(
                tf.reshape(
                    data,
                    [
                        rows,
                        tf.shape(data)[-3] // rows,
                        cols,
                        tf.shape(data)[-2] // cols,
                        channels,
                        tf.shape(data)[-1] // channels,
                    ],
                ),
                axis=1,
            ),
            axis=2,
        ),
        axis=3,
    )


def tf_extract_patches(tf_array, ksize):
    """
    Extract Patches from `tf_array`.

    Other implementation of `tf.image.extract_patches`.

    Conditions.
        * Padding is "SAME".
        * Stride is 1.
        * Width and Height are equal.

    Parameters
    ----------
    tf_array : `Tensor`
        Tensor array to extract patches.
    ksize : int
        Should be odd integer.

    Returns
    -------
    `Tensor`
        Patch extracted `tf_array`
    """
    padding_size = max((ksize - 1), 0) // 2
    zero_padded_image = tf.keras.layers.ZeroPadding2D((padding_size, padding_size))(
        tf_array
    )

    batch_size = tf.shape(tf_array)[0]
    img_wh = tf.shape(tf_array)[1]
    channel = tf.shape(tf_array)[-1]

    wh_indices = tf.range(ksize) + tf.range(img_wh)[:, tf.newaxis]

    a0 = tf.reshape(
        tf.repeat(tf.range(batch_size), img_wh * img_wh * ksize * ksize),
        (batch_size * img_wh * img_wh, ksize * ksize),
    )
    a1 = tf.tile(
        tf.repeat(tf.repeat(wh_indices, ksize, axis=1), img_wh, axis=0), (batch_size, 1)
    )
    a2 = tf.tile(tf.tile(wh_indices, (img_wh, ksize)), (batch_size, 1))

    m = tf.stack([a0, a1, a2], axis=-1)
    m2 = tf.reshape(m, (batch_size, img_wh, img_wh, ksize, ksize, 3))

    gg = tf.gather_nd(zero_padded_image, m2, batch_dims=0)
    gg2 = tf.reshape(gg, (batch_size, img_wh, img_wh, ksize * ksize * channel))
    return gg2
