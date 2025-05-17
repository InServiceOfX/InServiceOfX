from corecode.Utilities import load_environment_file
from pathlib import Path

from morelumaai.Configuration import GenerationConfiguration
from morelumaai.Wrappers.ImageAndVideoManager import ImageAndVideoManager

load_environment_file()
test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_ImageAndVideoManager_update_generations_list():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    image_and_video_manager = ImageAndVideoManager(generation_configuration)

    assert image_and_video_manager._current_generations_list is None
    assert image_and_video_manager.parsed_generations == {}
    image_and_video_manager.update_generations_list()

    assert image_and_video_manager._current_generations_list is not None
    assert image_and_video_manager.parsed_generations != {}

    print("\n current_generations_list:")
    print(image_and_video_manager._current_generations_list)
    print("\n len(image_and_video_manager._current_generations_list.generations):")
    print(len(image_and_video_manager._current_generations_list.generations))
    print("\n parsed_generations:")
    print(image_and_video_manager.parsed_generations)
    print("\n len(image_and_video_manager.parsed_generations):")
    print(len(image_and_video_manager.parsed_generations))

def test_ImageAndVideoManager_create_formatted_generations_list():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    image_and_video_manager = ImageAndVideoManager(generation_configuration)
    image_and_video_manager.update_generations_list()

    formatted_generations_list = \
        image_and_video_manager._create_formatted_generations_list()

    print("\n formatted_generations_list:")
    for (index, generation) in enumerate(formatted_generations_list):
        print(index)
        print(generation)
