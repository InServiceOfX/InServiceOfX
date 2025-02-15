from corecode.Utilities import load_environment_file
from pathlib import Path

from morelumaai.Configuration import GenerationConfiguration
from morelumaai.Wrappers.ImageAndVideoManager import (
    ImageAndVideoManager,
    ImageFrame)

load_environment_file()
test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

prompt1 = (
    "A young, beautiful, gorgeous blonde woman poses for a Static camera, "
    "subtly shifting poses only so slightly after 2-3 seconds pausing "
    "statically on each pose in a Static background.")


def test_ImageAndVideoManager_can_generate_video_from_start_frame():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    image_and_video_manager = ImageAndVideoManager(generation_configuration)

    image_and_video_manager.add_image(
        "https://pbs.twimg.com/media/GjyaI2RacAAj94v?format=jpg",
        "Human figure 0")

    assert image_and_video_manager.available_images == [
        ImageFrame(
            url="https://pbs.twimg.com/media/GjyaI2RacAAj94v?format=jpg",
            prompt_description="Human figure 0"
        )]

    image_and_video_manager.set_start_frame(
        image_and_video_manager.available_images[0])
    assert image_and_video_manager.end_frame == None
    assert image_and_video_manager.start_frame == \
        image_and_video_manager.available_images[0]

    print(
        "Before, all generations:",
        image_and_video_manager.generate_video.list_all_generations())

    generate_result = image_and_video_manager.generate(prompt1)
    print(generate_result)
    print(image_and_video_manager.generate_video.current_generation)
    print(type(image_and_video_manager.generate_video.current_generation))

    print(
        "After, all generations:",
        image_and_video_manager.generate_video.list_all_generations())

def test_ImageAndVideoManager_can_generate_video_from_end_frame():
    """
    https://docs.lumalabs.ai/docs/python-video-generation#with-ending-frame
    """
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    image_and_video_manager = ImageAndVideoManager(generation_configuration)

    assert image_and_video_manager.available_images == []
    assert len(image_and_video_manager.available_generations) == 0

    image_and_video_manager.add_image(
        "https://pbs.twimg.com/media/GjyaI2RacAAj94v?format=jpg",
        "Human figure 0")

    assert image_and_video_manager.available_images == [
        ImageFrame(
            url="https://pbs.twimg.com/media/GjyaI2RacAAj94v?format=jpg",
            prompt_description="Human figure 0"
        )]

    image_and_video_manager.set_end_frame(
        image_and_video_manager.available_images[0])
    assert image_and_video_manager.start_frame == None
    assert image_and_video_manager.end_frame == \
        image_and_video_manager.available_images[0]

    generate_result = image_and_video_manager.generate(prompt1)
    print(generate_result)
    print(image_and_video_manager.generate_video.current_generation)
    print(type(image_and_video_manager.generate_video.current_generation))

def test_ImageAndVideoManager_cannot_generate_with_start_and_end_keyframes_for_ray2():
    """
    https://docs.lumalabs.ai/docs/python-video-generation#with-start-and-end-keyframes
    """
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    generation_configuration.model = "ray-1-6"
    # lumaai.BadRequestError: Error code: 400 - {'detail': '400: Duration is not supported in Ray 1.6 API'}
    generation_configuration.duration = None
    # lumaai.BadRequestError: Error code: 400 - {'detail': '400: Resolution is not supported in Ray 1.6 API'}
    generation_configuration.resolution = None

    image_and_video_manager = ImageAndVideoManager(generation_configuration)

    image_and_video_manager.add_image(
        "https://pbs.twimg.com/media/GjzK1IFbsAAKbz3?format=jpg",
        "Human figure 0")

    image_and_video_manager.add_image(
        "https://pbs.twimg.com/media/GjzK1H5bcAAvRxl?format=jpg",
        "Human figure 1")

    image_and_video_manager.set_start_frame(
        image_and_video_manager.available_images[0])

    image_and_video_manager.set_end_frame(
        image_and_video_manager.available_images[1])

    prompt = (
        "Tall and beautiful 25 year old woman, long blond hair, standing "
        "in a high class hotel room, poses for a camera that slowly Orbit Left "
        "around in as the young woman makes static poses for the camera that "
        "stops slowly to Static pose and Static background.")

    generate_result = image_and_video_manager.generate(prompt)
    print(generate_result)
    print(image_and_video_manager.generate_video.current_generation)
    print(type(image_and_video_manager.generate_video.current_generation))

def test_ImageAndVideoManager_cannot_extend_video_with_ray2():
    """
    https://docs.lumalabs.ai/docs/python-video-generation#extend-video
    """
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    # lumaai.BadRequestError: Error code: 400 - {'detail': '400: Ray 2 with extend or interpolate is not available yet'}
    generation_configuration.model = "ray-1-6"
    generation_configuration.duration = None
    generation_configuration.resolution = None

    image_and_video_manager = ImageAndVideoManager(generation_configuration)

    image_and_video_manager.update_generations_list()

    generation_id = "c93374e9-b9ff-425a-a588-782d54c7d651"
    image_and_video_manager.set_start_frame(
        image_and_video_manager._get_generation_by_id(generation_id))

    prompt = (
        "23 year old Swedish woman, stunning beauty, very long wavy blonde "
        "hair, thin, athletic, wearing white lacy apron with red and green "
        "flower design, blue eyes, looking at viewer, camera is Static and "
        "does not move, while woman is static and does not move as she "
        "slowly sips her the coffee hidden inside her cup."
    )
    # lumaai.BadRequestError: Error code: 400 - {'detail': '400: Ray 2 with extend or interpolate is not available yet'}
    generate_result = image_and_video_manager.generate(prompt)
    print(generate_result)
    print(image_and_video_manager.generate_video.current_generation)
    # <class 'lumaai.types.generation.Generation'>
    print(type(image_and_video_manager.generate_video.current_generation))

def test_ImageAndVideoManager_cannot_reverse_extend_video_with_ray2():
    """
    https://docs.lumalabs.ai/docs/python-video-generation#reverse-extend-video
    """
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    # lumaai.BadRequestError: Error code: 400 - {'detail': '400: Ray 2 with extend or interpolate is not available yet'}
    generation_configuration.model = "ray-1-6"
    generation_configuration.duration = None
    generation_configuration.resolution = None

    image_and_video_manager = ImageAndVideoManager(generation_configuration)

    image_and_video_manager.update_generations_list()

    generation_id = "23097a97-a3f2-4685-9bf9-8f9b1f0f987d"
    image_and_video_manager.set_end_frame(
        image_and_video_manager._get_generation_by_id(generation_id))

    prompt = (
        "Beautiful gorgeous all-American Midwestern blonde American co-ed "
        "country girl posing for a\nStatic camera only demurly and subtly "
        "grinning, keeping hands and arms to her sides pressed statically "
        "and standing against a barn wall to pose sensually for the camera "
        "on a hot summer sizzling day")

    # lumaai.BadRequestError: Error code: 400 - {'detail': '400: Ray 2 with extend or interpolate is not available yet'}
    generate_result = image_and_video_manager.generate(prompt)
    print(generate_result)
    print(image_and_video_manager.generate_video.current_generation)
    # <class 'lumaai.types.generation.Generation'>
    print(type(image_and_video_manager.generate_video.current_generation))

def test_ImageAndVideoManager_can_delete_generation():
    generation_configuration = GenerationConfiguration.from_yaml(
        test_data_directory / "generation_configuration.yml")

    image_and_video_manager = ImageAndVideoManager(generation_configuration)

    image_and_video_manager.update_generations_list()

    formatted_generations_list = \
        image_and_video_manager._create_formatted_generations_list()

    # # 41
    # print("\n len(formatted_generations_list):")
    # print(len(formatted_generations_list))
    # print("\n formatted_generations_list:")
    # for (index, generation) in enumerate(formatted_generations_list):
    #     print(index)
    #     print(generation)
    
    # # Returns a None type.
    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "887b55bf-44c6-4eb6-b69b-a006e23f8a92")
    # print(deleted_generation)
    # # This should be None.
    # print(type(deleted_generation))

    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "9a205753-3def-4722-86ee-d43257ba50da")

    # print(deleted_generation)
    # print(type(deleted_generation))

    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "1b916336-c626-4a0b-b4b7-0e819ea40d0c")

    # print(deleted_generation)
    # print(type(deleted_generation))

    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "bf946bbb-5795-4e46-a0c5-7626b42454f9")

    # print(deleted_generation)
    # print(type(deleted_generation))

    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "14494a8b-3cd6-48ea-a475-d17629ba25ec")

    # print(deleted_generation)
    # print(type(deleted_generation))

    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "98409316-4c32-40e8-a726-dcc6a689dd33")

    # print(deleted_generation)
    # print(type(deleted_generation))

    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "e2e735ca-fffa-460f-83d9-164eed3cabdd")

    # print(deleted_generation)
    # print(type(deleted_generation))

    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "656871a7-e6c6-415b-b836-423fa12647cd")

    # print(deleted_generation)
    # print(type(deleted_generation))

    # deleted_generation = \
    #     image_and_video_manager.generate_video.delete_generation(
    #         "15b0452d-9cab-489c-95a7-ae6f948c6b7f")

    # print(deleted_generation)
    # print(type(deleted_generation))


    # image_and_video_manager.update_generations_list()

    # formatted_generations_list = \
    #     image_and_video_manager._create_formatted_generations_list()

    # # 32
    # print("\n len(formatted_generations_list):")
    # print(len(formatted_generations_list))
    # print("\n formatted_generations_list:")
    # for (index, generation) in enumerate(formatted_generations_list):
    #     print(index)
    #     print(generation)

    # image_and_video_manager.generate_video.delete_generation(
    #     "73d4ff05-b310-467a-8d37-13780f15256f")

    # image_and_video_manager.generate_video.delete_generation(
    #     "b0be9dd1-8d5e-4536-ae6c-292bb47a0997")

    # image_and_video_manager.generate_video.delete_generation(
    #     "ccd1c047-391e-4325-a107-bf3933a247db")

    # image_and_video_manager.generate_video.delete_generation(
    #     "ece2c0e9-b020-4740-9218-10d965eb914a")

    # image_and_video_manager.generate_video.delete_generation(
    #     "361b45ee-2457-473e-a374-abdffc626d5d")
    
    # image_and_video_manager.generate_video.delete_generation(
    #     "fca4bf63-6601-4625-9a95-1a07ec36aa06")

    # image_and_video_manager.generate_video.delete_generation(
    #     "ca325b5f-4908-4bb5-b35a-0167787a6282")

    image_and_video_manager.generate_video.delete_generation(
        "142cf5e2-0a57-4218-b245-01f05e67ca8c")        

    image_and_video_manager.generate_video.delete_generation(
        "786923bd-bc93-4da5-a4c5-3d84441ab7ed")

    image_and_video_manager.update_generations_list()

    formatted_generations_list = \
        image_and_video_manager._create_formatted_generations_list()

    print("\n len(formatted_generations_list):")
    print(len(formatted_generations_list))
    print("\n formatted_generations_list:")
    for (index, generation) in enumerate(formatted_generations_list):
        print(index)
        print(generation)
