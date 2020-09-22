from pathlib import Path
from typing import List, Optional, Tuple

from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator


class FlowFromDirectory:
    """
    디렉토리에서 이미지를 읽어오는 keras의 `ImageDataGenerator`의 `flow_from_directory()` 메소드를 보조합니다.
    """

    def __init__(
        self,
        from_directory: str,
        class_directorys: List[str],
        batch_size: int,
        color_mode: str = "rgb",
        target_size: Tuple[int, int] = (256, 256),
        interpolation: str = "nearest",
        save_to_dir: Optional[str] = None,
        save_format: str = "png",
        save_prefix: str = "",
        shuffle: bool = True,
        seed: int = 42,
        class_mode: Optional[str] = None,
        subset: Optional[str] = None,
        follow_links: bool = False,
    ):
        self.from_directory = from_directory
        self.class_directorys = class_directorys
        self.batch_size = batch_size
        self.color_mode = color_mode
        self.target_size = target_size
        self.interpolation = interpolation
        self.save_to_dir = save_to_dir
        self.save_format = save_format
        self.save_prefix = save_prefix
        self.shuffle = shuffle
        self.seed = seed
        self.class_mode = class_mode
        self.subset = subset
        self.follow_links = follow_links

    def get_iterator(
        self, generator: ImageDataGenerator = ImageDataGenerator()
    ) -> DirectoryIterator:
        """
        디렉토리에서 이미지를 읽어오는 keras의 `ImageDataGenerator`의 `flow_from_directory()` 메소드를 보조합니다.

        Parameters
        ----------
        generator : ImageDataGenerator, optional
            데이터 증식를 위한 `ImageDataGenerator`를 직접 지정할 수 있습니다, by default ImageDataGenerator()

        Returns
        -------
        DirectoryIterator
            디렉토리로부터 만들어진 `ImageDataGenerator`
        """
        return generator.flow_from_directory(
            self.from_directory,
            classes=self.class_directorys,
            target_size=self.target_size,
            batch_size=self.batch_size,
            color_mode=self.color_mode,
            class_mode=self.class_mode,
            shuffle=self.shuffle,
            seed=self.seed,
            save_to_dir=self.save_to_dir,
            save_prefix=self.save_prefix,
            save_format=self.save_format,
            follow_links=self.follow_links,
            subset=self.subset,
            interpolation=self.interpolation,
        )


class ImagesFromDirectory(FlowFromDirectory):
    """
    디렉토리로부터 이미지를 읽어오는 클래스

    keras의 `ImageDataGenerator`의 `flow_from_directory()` 메소드를 보조하려는 목적입니다.
    폴더는 단일 클래스로 인식됩니다. (이미지 세그먼테이션 용도)
    """

    def __init__(
        self,
        dataset_directory: str,
        batch_size: int,
        color_mode: str = "rgb",
        target_size: Tuple[int, int] = (256, 256),
        interpolation: str = "nearest",
        save_to_dir: Optional[str] = None,
        save_format: str = "png",
        save_prefix: str = "",
        shuffle: bool = True,
        seed: int = 42,
        class_mode: Optional[str] = None,
        subset: Optional[str] = None,
        follow_links: bool = False,
    ):
        base_folder: str = str(Path(dataset_directory).parent)
        dataset_folder: str = str(Path(dataset_directory).stem)
        super().__init__(
            from_directory=base_folder,
            class_directorys=[dataset_folder],
            batch_size=batch_size,
            color_mode=color_mode,
            target_size=target_size,
            interpolation=interpolation,
            save_to_dir=save_to_dir,
            save_format=save_format,
            save_prefix=save_prefix,
            shuffle=shuffle,
            seed=seed,
            class_mode=class_mode,
            subset=subset,
            follow_links=follow_links,
        )
