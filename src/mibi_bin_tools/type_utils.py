from typing import Union, Iterable
from alpineer import misc_utils


def any_true(a: Union[bool, Iterable[bool]]) -> bool:
    """ `any` that allows singleton values

    Args:
        a (bool | Iterable[bool]):
            value or iterable of booleans

    Returns:
        bool:
            whether any true values where found
    """
    return any(misc_utils.make_iterable(a))
