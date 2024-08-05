def format_float_for_string(value, decimal_places=3):
    try:
        if value == int(value):
            return f"{int(value)}"
        else:
            # Truncate to 3 places, remove trailing zeros
            return f"{value:.{decimal_places}f}".rstrip('0').rstrip('.')
    except TypeError:
        input_value = value.value
        if input_value == int(input_value):
            return f"{int(input_value)}"
        else:
            return f"{input_value:.{decimal_places}f}".rstrip('0').rstrip('.')
