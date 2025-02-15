from corecode.Utilities import load_environment_file
from pathlib import Path

from lumaai.types import GenerationListResponse

from morelumaai.Configuration import GenerationConfiguration
from morelumaai.Wrappers import GenerateVideo

load_environment_file()
test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_GenerateVideo_changes_configuration():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    video_generator = GenerateVideo(generation_configuration)

    assert video_generator.configuration.loop is None

    video_generator.set_loop(True)

    assert video_generator.configuration.loop is True

    video_generator.set_loop(False)

    assert video_generator.configuration.loop is None

def test_GenerateVideo_can_list_all_generations():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    video_generator = GenerateVideo(generation_configuration)

    listed_generations = video_generator.list_all_generations()

    # <class 'lumaai.types.generation_list_response.GenerationListResponse'>
    # print(type(listed_generations))

    assert type(listed_generations) == GenerationListResponse
    print(listed_generations)
