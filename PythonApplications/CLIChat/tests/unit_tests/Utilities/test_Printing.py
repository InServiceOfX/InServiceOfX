import pytest
from clichat.Utilities.Printing import Printing
from clichat.Configuration import Configuration
from unittest.mock import patch
from collections import namedtuple
from pathlib import Path

# Create a terminal size tuple for mocking
TerminalSize = namedtuple('TerminalSize', ['columns', 'lines'])

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

@pytest.fixture
def config():
    test_file_path = test_data_directory / "clichat_configuration.yml"
    assert test_file_path.exists()
    return Configuration(test_file_path)

@pytest.fixture
def printer(config):
    return Printing(config)

def test_print_wrapped_text_with_wrapping(printer):
    with patch('builtins.print') as mock_print, \
         patch('shutil.get_terminal_size') as mock_size:
        mock_size.return_value = TerminalSize(columns=20, lines=24)
        
        printer.print_wrapped_text("This is a long text that should be wrapped")
        
        # Should be wrapped due to terminal width of 20
        mock_print.assert_called_once()
        called_text = mock_print.call_args[0][0]
        assert "This is a long text" in called_text
        assert "that should be" in called_text
        assert "wrapped" in called_text

def test_print_wrapped_text_without_wrapping(config, printer):
    config.wrap_words = False
    with patch('builtins.print') as mock_print:
        text = "This text should not be wrapped"
        printer.print_wrapped_text(text)
        mock_print.assert_called_once_with(text)

def test_print_as_html_formatted_text(printer):
    with patch('clichat.Utilities.Printing.print_as_html_formatted_text') \
        as mock_print:
        printer.print_as_html_formatted_text("Test content")
        mock_print.assert_called_once()
        assert "Test content" in mock_print.call_args[0][0]

def test_print_key_value_with_valid_format(printer):
    with patch('clichat.Utilities.Printing.print_key_value') as mock_print:
        printer.print_key_value("key: value")
        mock_print.assert_called_once()
        html_content = mock_print.call_args[0][0]
        assert "key:" in html_content
        assert "value" in html_content

def test_print_key_value_with_invalid_format(printer):
    with patch('clichat.Utilities.Printing.print_key_value') as mock_print:
        printer.print_key_value("invalid_format")
        mock_print.assert_called_once()
        html_content = mock_print.call_args[0][0]
        assert "invalid_format" in html_content