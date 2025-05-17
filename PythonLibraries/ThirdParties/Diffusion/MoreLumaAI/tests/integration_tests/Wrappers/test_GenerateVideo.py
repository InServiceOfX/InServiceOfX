from corecode.Utilities import load_environment_file
from pathlib import Path

from morelumaai.Configuration import GenerationConfiguration
from morelumaai.Wrappers import GenerateVideo

import requests

load_environment_file()
test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_GenerateVideo_can_generate_video():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    prompt = (
        "Aerial wide shot pushing in on a New England sprawling estate on a lake at "
        "dusk, with elegant gardens and elegant guests mingling on a terrace near the "
        "water circa 1930s.")

    video_generator = GenerateVideo(generation_configuration)

    # https://docs.lumalabs.ai/docs/python-video-generation

    keyframes = {
        "frame0": {
            "type": "image",
            "url": "https://i.imgur.com/TPo45xW.jpeg"
        }
    }

    video_url = video_generator.generate(prompt, keyframes) 
    
    save_path = Path(generation_configuration.temporary_save_path) / \
        f"{video_generator.current_generation.id}.mp4"

    # download the video
    response = requests.get(video_url, stream=True)
    with open(str(save_path), 'wb') as file:
        file.write(response.content)
    print(f"File downloaded as {save_path}")

def test_GenerateVideo_can_save_video():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    prompt = (
        "Young Swedish woman slowly drinking coffee from her cup in a still "
        "and quiet kitchen while the steam is slowly rising from her cup and "
        "her long blonde beautiful hair is mostly still and waving only so "
        "slightly.")

    video_generator = GenerateVideo(generation_configuration)

    # https://docs.lumalabs.ai/docs/python-video-generation

    keyframes = {
        "frame0": {
            "type": "image",
            "url": "https://i.imgur.com/5rJtsUk.png"
        }
    }

    video_url = video_generator.generate(prompt, keyframes) 

    save_path = video_generator.save_video()

    print(f"Video saved to {save_path}")

def test_GenerateVideo_can_save_video_2():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    # prompt = (
    #     "Aerial wide shot of a New England sprawling estate on a lake, late "
    #     "afternoon, with elegant gardens and elegant guests mingling on a "
    #     "terrace near the water circa 1930s.")

    prompt = (
        "75 years later: Aerial wide shot of a dilapidated New England "
        "sprawling estate on a lake, late afternoon, with overgrown gardens "
        "on a terrace near the water after decades of decline.")

    video_generator = GenerateVideo(generation_configuration)

    # https://docs.lumalabs.ai/docs/python-video-generation

    keyframes = {
        "frame0": {
            "type": "image",
            "url": "https://i.imgur.com/lAccpr8.jpeg"
        }
    }

    video_url = video_generator.generate(prompt, keyframes) 

    save_path = video_generator.save_video()

    print(f"Video saved to {save_path}")


