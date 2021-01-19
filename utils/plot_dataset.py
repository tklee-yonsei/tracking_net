from typing import Callable, List, Tuple

import tensorflow as tf


def take_from_dataset_at_all_batch(
    dataset, inout_processing_funcs: Tuple[List[Callable]] = ([], [])
):
    input_images = []
    output_images = []

    for elem in dataset.take(1):
        batch_size = tf.shape(elem[0][0])[0]

        inputs = elem[0]
        input_processings = inout_processing_funcs[0]
        if len(input_processings) == 1:
            input_b = []
            for b_index in range(batch_size):
                img = input_processings[0](inputs[b_index])
                input_b.append(img)
            input_images.append(input_b)
        elif len(input_processings) > 1:
            for input_index, input in enumerate(inputs):
                input_b = []
                for b_index in range(batch_size):
                    img = input_processings[input_index](input[b_index])
                    input_b.append(img)
                input_images.append(input_b)

        outputs = elem[1]
        output_processings = inout_processing_funcs[1]
        if len(output_processings) == 1:
            output_b = []
            for b_index in range(batch_size):
                img = output_processings[0](outputs[b_index])
                output_b.append(img)
            output_images.append(output_b)
        elif len(output_processings) > 1:
            for output_index, output in enumerate(outputs):
                output_b = []
                for b_index in range(batch_size):
                    img = output_processings[output_index](output[b_index])
                    output_b.append(img)
                output_images.append(output_b)

    return input_images, output_images
