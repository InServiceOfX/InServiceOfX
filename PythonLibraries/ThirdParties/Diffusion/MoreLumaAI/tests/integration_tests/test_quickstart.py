from corecode.Utilities import load_environment_file
from pathlib import Path
import requests
import time

from lumaai import LumaAI
from morelumaai.Configuration import GenerationConfiguration

load_environment_file()
test_data_directory = Path(__file__).resolve().parents[1] / "TestData"

def test_LumaAI_inits():
    client = LumaAI()
    assert isinstance(client, LumaAI)

def test_LumaAI_can_generate_video():
    """
    https://docs.lumalabs.ai/docs/python-video-generation#how-do-i-get-the-video-for-a-generation
    """
    # Fails on this long prompt.
    # prompt = (
    #     "A grand 19th-century estate stands majestically in the golden light of late "
    #     "afternoon, surrounded by lush greenery. The stately mansion, adorned with tall "
    #     "windows, intricate architectural details, and elegant balconies, overlooks a "
    #     "tranquil pond reflecting the warm hues of the sky. A grand gathering unfolds on "
    #     "the manicured lawn, with elegantly dressed guests in period attire—women in "
    #     "flowing cream-colored gowns and bonnets, men in dark suits and top hats—"
    #     "engaging in lively conversation. Some guests stand near the water's edge, while "
    #     "others enjoy the ambiance under large white parasols on the terrace. The "
    #     "golden-hour light bathes the scene in a warm glow, casting long shadows across "
    #     "the grass, with towering trees framing the picturesque setting. The mood is "
    #     "refined, nostalgic, and cinematic, evoking the elegance of a bygone era."
    # )

    # # Another to possibly try:
    # prompt = (
    #     "water flowing out of an enchanted gateway mirror, portal to another world, "
    #     "desert sand flying in the desolate wind, lush verdant landscape inside the "
    #     "mirror"
    # )

    # prompt = (
    #     "Aerial wide shot pushing in on a New England sprawling estate on a lake at "
    #     "dusk, with elegant gardens and elegant guests mingling on a terrace near the "
    #     "water circa 1940s."
    # )

    prompt = (
        "Aerial wide shot pushing in on a New England sprawling estate on a lake at "
        "dusk, with elegant gardens and elegant guests mingling on a terrace near the "
        "water circa 1930s."
    )

    client = LumaAI()
    generation = client.generations.create(
        prompt=prompt,
    )
    start_time = time.time()
    completed = False
    while not completed:
        generation = client.generations.get(id=generation.id)
        if generation.state == "completed":
            completed = True
        elif generation.state == "failed":
            raise RuntimeError(
                f"Generation failed: {generation.failure_reason}")
        elapsed_time = time.time() - start_time
        print(f"State: {generation.state} - Time elapsed: {elapsed_time:.2f}s")
        time.sleep(3)

    total_time = time.time() - start_time
    print(f"Total generation time: {total_time:.2f}s")

    video_url = generation.assets.video

    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    # Create save path using configuration
    save_path = Path(generation_configuration.temporary_save_path) / \
        f"{generation.id}.mp4"

    # download the video
    response = requests.get(video_url, stream=True)
    with open(str(save_path), 'wb') as file:
        file.write(response.content)
    print(f"File downloaded as {save_path}")

def test_LumaAI_can_generate_video_with_GenerationConfiguration():
    prompt = (
        "water flowing out of an enchanted gateway mirror, portal to another world, "
        "desert sand flying in the desolate wind, lush verdant landscape inside the "
        "mirror"
    )

    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    client = LumaAI()
    generation = client.generations.create(
        prompt=prompt,
        **generation_configuration.to_api_kwargs()
    )

    start_time = time.time()
    completed = False
    while not completed:
        generation = client.generations.get(id=generation.id)
        if generation.state == "completed":
            completed = True
        elif generation.state == "failed":
            raise RuntimeError(
                f"Generation failed: {generation.failure_reason}")
        elapsed_time = time.time() - start_time
        print(f"State: {generation.state} - Time elapsed: {elapsed_time:.2f}s")
        time.sleep(3)

    total_time = time.time() - start_time
    print(f"Total generation time: {total_time:.2f}s")

    video_url = generation.assets.video

    save_path = Path(generation_configuration.temporary_save_path) / \
        f"{generation.id}.mp4"

    # download the video
    response = requests.get(video_url, stream=True)
    with open(str(save_path), 'wb') as file:
        file.write(response.content)
    print(f"File downloaded as {save_path}")