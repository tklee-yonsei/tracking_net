from typing import Callable, List, Tuple

import numpy as np
import pylab
import tensorflow as tf
from matplotlib import pyplot as plt


def take_from_dataset_at_all_batch(
    dataset, inout_processing_funcs: Tuple[List[Callable]] = ([], [])
):
    input_images = []
    output_images = []

    for elem in dataset.take(1):
        batch_size = tf.shape(elem[0][0])[0]

        inputs = elem[0]
        input_processing_funcs = inout_processing_funcs[0]
        outputs = elem[1]
        output_processings_funcs = inout_processing_funcs[1]

        for b_index in range(batch_size):
            input_b = []
            if len(input_processing_funcs) == 1:
                img = input_processing_funcs[0](inputs[b_index])
                input_b.append(img)
            elif len(input_processing_funcs) > 1:
                for input_index, input in enumerate(inputs):
                    img = input_processing_funcs[input_index](input[b_index])
                    input_b.append(img)

            output_b = []
            if len(output_processings_funcs) == 1:
                img = output_processings_funcs[0](outputs[b_index])
                output_b.append(img)
            elif len(output_processings_funcs) > 1:
                for output_index, output in enumerate(outputs):
                    img = output_processings_funcs[output_index](output[b_index])
                    output_b.append(img)

            input_images.append(input_b)
            output_images.append(output_b)

    return input_images, output_images


def plot_samples(images, save_file_name, element_width=3, element_height=3):
    r = len(images)
    c = max(list(map(lambda el: len(el), images)))

    f, axarr = plt.subplots(r, c, figsize=(element_width * c, element_height * r))
    f.patch.set_facecolor("xkcd:white")
    for r_i in range(r):
        for c_i in range(c):
            axarr[r_i, c_i].xaxis.set_ticks([])
            axarr[r_i, c_i].yaxis.set_ticks([])
            axarr[r_i, c_i].axis("off")

    pylab.gray()

    for row_index, image_row in enumerate(images):
        for col_index, image_col in enumerate(image_row):
            axarr[row_index, col_index].imshow(np.squeeze(image_col))

    f.savefig(save_file_name)
