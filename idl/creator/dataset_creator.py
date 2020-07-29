import abc


def check_hasattr(cls, subclass) -> bool:
    import inspect
    from typing import List

    cls_function_names: List[str] = list(
        map(lambda el: el[0], inspect.getmembers(cls, predicate=inspect.isfunction))
    )
    return all(
        hasattr(subclass, cls_function_name) for cls_function_name in cls_function_names
    )


class DatasetCreator(metaclass=abc.ABCMeta):
    """
    [Interface]
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            check_hasattr(cls, subclass)
            and callable(subclass.load_data_source)
            or NotImplemented
        )

    @abc.abstractmethod
    def load_data_source(self, path: str, file_name: str):
        raise NotImplementedError


class BaseDatasetCreator(DatasetCreator):
    def load_data_source(self, path: str, file_name: str):
        print(path)


class TileDatasetCreator(DatasetCreator, metaclass=abc.ABCMeta):
    """
    [Interface]
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return check_hasattr(cls, subclass) and callable(subclass.aa) or NotImplemented

    @abc.abstractmethod
    def aa(self, path: str, file_name: str):
        raise NotImplementedError


class BaseTileDatasetCreator(
    BaseDatasetCreator, TileDatasetCreator, metaclass=abc.ABCMeta
):
    def aa(self, path: str, file_name: str):
        print(file_name)
