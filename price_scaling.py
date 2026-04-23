"""
price_scaling.py
----------------
Converts raw Lightstreamer / PDS values to scaled prices.

SR3 uses ÷100 → basis points display
"""

_OPTION_SCALING: dict[str, dict[str, callable]] = {
    "SR3": {
        "underlying": lambda x: x / 100,
        "strike":     lambda x: x / 100,
        "bid":        lambda x: x / 100,
        "ask":        lambda x: x / 100,
    },
}

_FUTURE_SCALING: dict[str, dict[str, callable]] = {
    "SR3": {
        "bid":  lambda x: x / 100,
        "ask":  lambda x: x / 100,
    },
}

_DEFAULT_OPTION: dict[str, callable] = {
    "underlying": lambda x: (x / 1),
    "strike":     lambda x: (x / 1),
    "bid":        lambda x: (x / 1),
    "ask":        lambda x: (x / 1),
}

_DEFAULT_FUTURE: dict[str, callable] = {
    "bid": lambda x: (x / 1),
    "ask": lambda x: (x / 1),
}


def scale(raw, family: str, field: str, is_option: bool) -> float | None:
    """
    Apply the lambda for (family, field) to raw and return the scaled float.
    Returns None if raw cannot be parsed as a number.
    """
    try:
        table   = _OPTION_SCALING if is_option else _FUTURE_SCALING
        default = _DEFAULT_OPTION if is_option else _DEFAULT_FUTURE
        fn      = table.get(family, default).get(field, lambda x: x)
        return fn(float(raw))
    except (ValueError, TypeError):
        return None