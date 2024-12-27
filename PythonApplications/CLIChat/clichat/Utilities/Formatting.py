import shutil, wcwidth, textwrap

def get_string_width(text: str) -> int:
    width = 0
    for character in text:
        width += wcwidth.wcwidth(character)
    return width

def wrap_text(content, terminal_width=None):
    if terminal_width is None:
        terminal_width = shutil.get_terminal_size().columns
    return "\n".join([
        textwrap.fill(line, width=terminal_width) \
            for line in content.splitlines()])

def empty_string_to_none(value):
    if value == "":
        return None
    return value