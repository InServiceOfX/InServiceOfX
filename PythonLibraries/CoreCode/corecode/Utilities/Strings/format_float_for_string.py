def format_float_for_string(value, decimal_places=3):
    if value == int(value):
        return f"{int(value)}"
    else:
        # Truncate to 3 places, remove trailing zeros
        return f"{value:.{decimal_places}f}".rstrip('0').rstrip('.')
