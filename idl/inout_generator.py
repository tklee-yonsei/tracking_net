import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Callable, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import toolz
from keras.preprocessing.image import ImageDataGenerator

from idl.batch_transform import generate_iterator_and_transform
from idl.flow_directory import FlowFromDirectory
from utils.generator import zip_generators
from utils.image_transform import InterpolationEnum, img_resize


def save_batch_transformed_img(
    target_folder: str,
    prefix: str,
    index_num: int,
    batch_num: int,
    transformed_batch_img: np.ndarray,
) -> None:
    # 이름
    img_name = "{}img_transformed_{:04d}_{:02d}.png".format(
        prefix, index_num, batch_num
    )
    img_fullpath = os.path.join(target_folder, img_name)

    # 저장
    cv2.imwrite(img_fullpath, transformed_batch_img)


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
        resize_to: Tuple[int, int],
        resize_interpolation: InterpolationEnum = InterpolationEnum.inter_nearest,
        image_data_generator: ImageDataGenerator = ImageDataGenerator(),
        image_transform_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        each_transformed_image_save_function_optional: Optional[
            Callable[[int, int, np.ndarray], None]
        ] = None,
        transform_function_for_all: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        디렉토리에서 파일을 불러오는 매니저입니다.

        Parameters
        ----------
        flow_from_directory : FlowFromDirectory
            [description]
        resize_to: Tuple[int, int]
            이미지를 리사이즈 할 크기를 지정합니다. (세로, 가로)
        resize_interpolation: InterpolationEnum
            Interpolation 정책을 설정합니다. by default InterpolationEnum.inter_nearest
        image_data_generator : ImageDataGenerator, optional
            [description], by default ImageDataGenerator()
        image_transform_function : Optional[Callable[[np.ndarray], np.ndarray]], optional
            배치 내 이미지 변환 함수. 변환 함수가 지정되지 않으면, 변환 없이 그냥 내보냅니다., by default None
        each_transformed_image_save_function_optional : Optional[Callable[[int, int, np.ndarray], None]], optional
            샘플 인덱스 번호, 배치 번호 및 이미지를 입력으로 하는 저장 함수, by default None
        transform_function_for_all : Optional[Callable[[np.ndarray], np.ndarray]], optional
            변환 함수. 배치 전체에 대한 변환 함수, by default None
        """
        self.flow_from_directory: FlowFromDirectory = flow_from_directory
        self.resize_to: Tuple[int, int] = resize_to
        self.image_data_generator: Optional[ImageDataGenerator] = image_data_generator
        _image_transform_function = lambda img: img
        if image_transform_function is not None:
            _image_transform_function = image_transform_function
        image_transform_function_with_resize = toolz.compose_left(
            lambda img: img_resize(img, resize_to, resize_interpolation),
            _image_transform_function,
        )
        self.image_transform_function = image_transform_function_with_resize
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

        if len(self.__output_flows) > 0:
            output_generators: List[Generator] = []
            for output_flow in self.__output_flows:
                o_generator = output_flow.flow_from_directory.get_iterator(
                    output_flow.image_data_generator
                )
                o_transformed_generator: Generator = generate_iterator_and_transform(
                    image_generator=o_generator,
                    each_image_transform_function=output_flow.image_transform_function,
                    each_transformed_image_save_function_optional=output_flow.each_transformed_image_save_function_optional,
                    transform_function_for_all=output_flow.transform_function_for_all,
                )
                output_generators.append(o_transformed_generator)
            outputs_generator = map(list, zip_generators(output_generators))

            return zip(inputs_generator, outputs_generator)
        else:
            return inputs_generator
