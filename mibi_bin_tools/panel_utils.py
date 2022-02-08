import pandas as pd


def make_panel(mass: float, target_name: str = None, low_range: float = 0.3,
               high_range: float = 0.0) -> pd.DataFrame:
    """ Creates single mass panel

    Args:
        mass (float):
            central m/z for signal
        target_name (str | None):
            naming for target. 'Target' if None
        low_range (float):
            units below central mass to start integration
        high_range (float):
            units above central mass to stop integration

    Returns:
        pd.DataFrame:
            single mass panel as pandas dataframe
    """
    return pd.DataFrame([{
        'Mass': mass,
        'Target': target_name or 'Target',
        'Start': mass - low_range,
        'Stop': mass + high_range,
    }])
