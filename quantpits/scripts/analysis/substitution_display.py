"""Display saved substitution returns without turning missing data into zero."""

import math


def finite_return(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def format_substitution_return(value, digits=4):
    number = finite_return(value)
    return "N/A" if number is None else format(number, ".%d%%" % digits)


def substitution_direction(value):
    number = finite_return(value)
    if number is None:
        return "Unknown"
    if number == 0:
        return "Neutral"
    return "Lucky/Gain" if number > 0 else "Unlucky/Loss"
