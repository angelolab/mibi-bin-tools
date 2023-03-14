from typing import Union, List
import pandas as pd

from alpineer import misc_utils


def make_panel(mass: Union[float, List[float]],
               target_name: Union[str, List[str], None] = None,
               low_range: Union[float, List[float]] = 0.3,
               high_range: Union[float, List[float]] = 0.0) -> pd.DataFrame:
    """ Creates single mass panel

    Args:
        mass (float | List[float]):
            central m/z for signal
        target_name (str | List[str] | None):
            naming for target. 'Target' if None
        low_range (float | List[float]):
            units below central mass to start integration
        high_range (float | List[float]):
            units above central mass to stop integration

    Returns:
        pd.DataFrame:
            single mass panel as pandas dataframe
    """

    mass = misc_utils.make_iterable(mass)
    if target_name is not None:
        target_name = misc_utils.make_iterable(target_name)
        if len(mass) != len(target_name):
            raise ValueError(
                '`mass` and `target_name` did not contain the same number of elements.  '
                'If target names aren\'t required, then set `target_name=None`.  '
            )
    else:
        target_name = [f'targ{i}' for i in range(len(mass))]

    # check for range lists
    for r in (low_range, high_range):
        if misc_utils.make_iterable(r) == r:
            if len(r) != len(mass):
                raise ValueError(
                    '`mass` and a range argument did not contain the same number of elements.  '
                    'If only one integration range is required, `low_range` and `high_range` can '
                    'be set to float values, e.g `low_range=0.3`'
                )

    low_range = misc_utils.make_iterable(low_range)
    high_range = misc_utils.make_iterable(high_range)

    if len(low_range) != len(mass):
        low_range = low_range * len(mass)
    if len(high_range) != len(mass):
        high_range = high_range * len(mass)

    rows = []
    for m, ch, low, high in zip(mass, target_name, low_range, high_range):
        rows.append({
            'Mass': m,
            'Target': ch,
            'Start': m - low,
            'Stop': m + high,
        })

    return pd.DataFrame(rows)
