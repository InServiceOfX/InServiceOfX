from morex.Utilities import parse_post_URL
import pytest

URL = "https://x.com/alex_prompter/status/1943232047738425642"

def test_parse_URL_steps():
    assert parse_post_URL(URL) == ("alex_prompter", "1943232047738425642")

def test_parse_post_URL_invalid_URL():
    with pytest.raises(ValueError):
        parse_post_URL("https://x.com/alex_prompter/stats/1943232047738425642")

