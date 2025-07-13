from commonapi.Messages import ParsePromptsCollection
from corecode.Utilities import DataSubdirectories

data_subdirectories = DataSubdirectories()

prompts_collection_path = data_subdirectories.PromptsCollection

def test_load_manually_copied_X_posts():
    parse_prompts_collection = ParsePromptsCollection(prompts_collection_path)
    lines_of_files = parse_prompts_collection.load_manually_copied_X_posts()
    assert lines_of_files is not None
    assert len(lines_of_files) == 1
    assert len(lines_of_files[0]) == 76

def test_parse_manually_copied_X_posts():
    parse_prompts_collection = ParsePromptsCollection(prompts_collection_path)
    lines_of_files = parse_prompts_collection.load_manually_copied_X_posts()
    posts = parse_prompts_collection.parse_manually_copied_X_posts(lines_of_files)
    assert posts is not None
    assert len(posts) == 8
    assert posts[0]['url'] == \
        'https://x.com/alex_prompter/status/1943232047738425642'
    assert posts[0]['section_title'] == \
        "1. Realistic Physics Game (Hexagon Test)"
    assert posts[0]['prompt'] == \
        (
            "Create a HTML, CSS, and javascript where a ball is inside a "
            "rotating hexagon. The ball is affected by Earth’s gravity and "
            "friction from the hexagon walls. The bouncing must appear "
            "realistic."
        )