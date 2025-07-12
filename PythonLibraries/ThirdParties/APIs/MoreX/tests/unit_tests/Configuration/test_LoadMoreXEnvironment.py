from pathlib import Path
from morex.Configuration import LoadMoreXEnvironment

test_data_dir = Path(__file__).parents[2] / "TestData"

def test_LoadMoreXEnvironment():
    load_morex_environment = LoadMoreXEnvironment(
        test_data_dir / "example_morex.env")
    load_morex_environment()
    assert load_morex_environment.get_environment_variable(
        "X_CONSUMER_API_KEY") == "a1"
    assert load_morex_environment.get_environment_variable(
        "X_CONSUMER_SECRET") == "bc23"
    assert load_morex_environment.get_environment_variable(
        "X_BEARER_TOKEN") == "def567"
    assert load_morex_environment.get_environment_variable(
        "X_ACCESS_TOKEN") == "ghij8901"
    assert load_morex_environment.get_environment_variable(
        "X_SECRET_TOKEN") == "klmno23456"