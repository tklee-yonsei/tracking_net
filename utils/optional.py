from typing import Optional, TypeVar

T = TypeVar("T")


def get_or_else(value_optional: Optional[T], default: T) -> T:
    """
    어떤 None이 될 수 있는 값 `value_optional`에 대해, 값이 존재하지 않는 경우 `default` 값을 리턴합니다.

    Parameters
    ----------
    value_optional : Optional[T]
        어떤 None이 될 수 있는 값
    default : T
        디폴드 값

    Returns
    -------
    T
        `value_optional` 값 혹은 디폴드 값
    """
    return default if value_optional is None else value_optional
