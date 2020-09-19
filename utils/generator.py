from functools import reduce
from typing import Generator, List


def zip_generators(generator_list: List[Generator]):
    """
    `generator_list` 들을 모아, 단일 Generator를 생성합니다.

    Parameters
    ----------
    generator_list : List[Generator]
        [description]

    Examples
    --------
    >>> def sample_generator(f=0, t=10):
    ...     for i in range(f, t):
    ...         yield i
    
    >>> sample_generator1 = sample_generator(0, 10)
    >>> sample_generator2 = sample_generator(5, 15)
    >>> sample_generator3 = sample_generator(16, 29)
    >>> sample_generator4 = sample_generator(30, 66)
    >>> sample_generators: List[Generator] = [sample_generator1, sample_generator2, sample_generator3, sample_generator4]
    >>> list_generator = map(list, zip_generators(sample_generators))
    >>> next(list_generator)
    [0, 5, 16, 30]
    >>> next(list_generator)
    [1, 6, 17, 31]
    """
    assert len(generator_list) != 0, "List is empty."

    def _tuple_reducer(a, b):
        for _element in zip(a, b):
            if isinstance(_element[0], tuple):
                yield _element[0] + (_element[1],)
            else:
                yield (_element[0],) + (_element[1],)

    if len(generator_list) == 1:
        return map(lambda el: (el,), generator_list[0])
    else:
        return reduce(lambda a, b: _tuple_reducer(a, b), generator_list)
