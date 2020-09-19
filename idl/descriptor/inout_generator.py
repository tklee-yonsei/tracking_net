from abc import ABC, ABCMeta, abstractmethod
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# from idl.utils import check_hasattr
from idl.batch_transform import generate_iterator_and_transform
from idl.flow_directory import FlowFromDirectory
from utils.generator import zip_generators

# import keras


class InOutGenerator(ABC, metaclass=ABCMeta):
    """
    [Interface]
    """

    @property
    @abstractmethod
    def input_flows(self) -> List[FlowFromDirectory]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_flows(self) -> List[FlowFromDirectory]:
        raise NotImplementedError


class FlowManager:
    def __init__(
        self,
        flow_from_directory: FlowFromDirectory,
        image_data_generator: ImageDataGenerator = ImageDataGenerator(),
        image_transform_function: Optional[
            Tuple[Callable[[np.ndarray], np.ndarray], Optional[Tuple[int, int, int]]]
        ] = None,
        each_transformed_image_save_function_optional: Optional[
            Callable[[int, int, np.ndarray], None]
        ] = None,
        transform_function_for_all: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.flow_from_directory: FlowFromDirectory = flow_from_directory
        self.image_data_generator: Optional[ImageDataGenerator] = image_data_generator
        self.image_transform_function = image_transform_function
        self.each_transformed_image_save_function_optional = (
            each_transformed_image_save_function_optional
        )
        self.transform_function_for_all = transform_function_for_all


class BaseInOutGenerator(InOutGenerator):
    def __init__(
        self, input_flows: List[FlowManager], output_flows: List[FlowManager] = [],
    ):
        self.__input_flows = input_flows
        self.__output_flows = output_flows
        self.__i_generator = None

    @property
    def input_flows(self) -> List[FlowFromDirectory]:
        return self.__input_flows

    @property
    def output_flows(self) -> List[FlowFromDirectory]:
        raise self.__output_flows

    def get_samples(self) -> int:
        return self.__i_generator.samples if self.__i_generator is not None else 0

    def get_filenames(self) -> List[str]:
        return self.__i_generator.filenames if self.__i_generator is not None else []

    def get_generator(self):
        input_generators: List[Generator] = []
        for index, input_flow in enumerate(self.__input_flows):
            i_generator = input_flow.flow_from_directory.get_iterator(
                input_flow.image_data_generator
            )
            if index == 0:
                self.__i_generator = i_generator
            i_transformed_generator: Generator = generate_iterator_and_transform(
                image_generator=i_generator,
                each_image_transform_function=input_flow.image_transform_function,
                each_transformed_image_save_function_optional=input_flow.each_transformed_image_save_function_optional,
                transform_function_for_all=input_flow.transform_function_for_all,
            )
            input_generators.append(i_transformed_generator)
        inputs_generator = map(list, zip_generators(input_generators))

        if len(self.__output_flows) != 0:
            output_generators: List[Generator] = []
            for output_flow in self.__output_flows:
                o_generator: Generator = generate_iterator_and_transform(
                    image_generator=output_flow.flow_from_directory.get_iterator(
                        output_flow.image_data_generator
                    ),
                    each_image_transform_function=output_flow.image_transform_function,
                    each_transformed_image_save_function_optional=output_flow.each_transformed_image_save_function_optional,
                    transform_function_for_all=output_flow.transform_function_for_all,
                )
                output_generators.append(o_generator)
            outputs_generator = map(list, zip_generators(output_generators))

            return zip(inputs_generator, outputs_generator)
        else:
            return inputs_generator
