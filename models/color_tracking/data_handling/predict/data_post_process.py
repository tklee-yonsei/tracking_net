from base.runnable import Runnable
from typing import Tuple


class TrackingPredictDataPostProcess(Runnable):
    """
    Tracking 네트워크의 Predict에 대한 추가 데이터 후처리
    """

    tile_size: Tuple[int, int]
    wrap_size: Tuple[int, int]

    def __init__(self, tile_size: Tuple[int, int], wrap_size: Tuple[int, int]):
        self.tile_size = tile_size
        self.wrap_size = wrap_size

    def merge(self, starts_with: str) -> None:
        pass
