import pytest
from clichat.Utilities.Formatting import get_string_width, wrap_text
import shutil
from unittest.mock import patch
from collections import namedtuple

# Create a terminal size tuple for mocking
TerminalSize = namedtuple('TerminalSize', ['columns', 'lines'])

def test_get_string_width_basic():
    text = "Hello World"
    assert get_string_width(text) == 11

def test_get_string_width_special_chars():
    # Test special ASCII characters
    text = "Hello\tWorld\n"
    assert get_string_width(text) == 8

def test_get_string_width_empty():
    assert get_string_width("") == 0

def test_get_string_width_spaces():
    text = "   "  # 3 spaces
    assert get_string_width(text) == 3

@pytest.mark.parametrize("terminal_width,expected", [
    (20, "This is a long\nsentence that needs\nto be wrapped."),
    (10, "This is a\nlong\nsentence\nthat needs\nto be\nwrapped.")
])
def test_wrap_text_with_width(terminal_width, expected):
    text = "This is a long sentence that needs to be wrapped."
    assert wrap_text(text, terminal_width) == expected

def test_wrap_text_multiline():
    text = "Line 1\nLine 2\nLine 3 is very long and should be wrapped"
    with patch('shutil.get_terminal_size') as mock_size:
        mock_size.return_value = TerminalSize(columns=20, lines=24)
        result = wrap_text(text)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3 is very" in result

def test_wrap_text_empty():
    assert wrap_text("") == ""

def test_wrap_text_no_wrap_needed():
    text = "Short line"
    with patch('shutil.get_terminal_size') as mock_size:
        mock_size.return_value = TerminalSize(columns=20, lines=24)
        assert wrap_text(text) == text