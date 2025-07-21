from corecode.FileIO import (
    get_file_and_directory_lists_recursively,
    get_project_directory_path,
)
from corecode.Utilities import DataSubdirectories, is_model_there
from corecode.Parsers.jinja import parse_jinja_template
import pytest

data_subdirectories = DataSubdirectories()

relative_model_path = "Models/LLM/HuggingFaceTB/SmolLM3-3B"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

model_is_not_downloaded_message = f"Model {relative_model_path} not downloaded"

@pytest.mark.skipif(
        not is_model_downloaded, reason=model_is_not_downloaded_message)
def test_parse_jinja_template():
    files, _ = get_file_and_directory_lists_recursively(
        model_path,
        get_project_directory_path() / ".gitignore"
    )

    chat_template_jinja_file_path = None
    for file in files:
        if ".jinja" in str(file) and "chat_template" in str(file):
            chat_template_jinja_file_path = file
            break

    assert chat_template_jinja_file_path is not None
    assert chat_template_jinja_file_path.exists()

    parsed_content = None

    with open(chat_template_jinja_file_path, "r") as f:
        chat_template_jinja_file_content = f.read()
        parsed_content = parse_jinja_template(chat_template_jinja_file_content)

    assert parsed_content is not None

    # Uncomment to print the parsed content
    #print("parsed_content", parsed_content)

    assert not parsed_content.get("is_valid")



