from corecode.Utilities import load_environment_file

load_environment_file()

import fal_client
import pytest
from fal_client import submit_async

def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
           print(log["message"])

def test_image_to_image_api_call():
    """
    https://fal.ai/models/fal-ai/flux/dev/image-to-image/api
    """
    # fal/projects/fal_client/src/fal_client/client.py def subscribe(..)
    application_name = "fal-ai/flux/dev/image-to-image"
    prompt = "A faded, old, a little burnt on the edges photo of a cat looking "
    prompt += "up at the camera and dressed as a wizard with a background of "
    prompt += "an American 1970s kitchen"
    image_url = "https://fal.media/files/koala/Chls9L2ZnvuipUTEwlnJC.png"

    result = fal_client.subscribe(
        application_name,
        arguments={
            "image_url": image_url,
            "prompt": prompt
        },
        with_logs=True,
        on_queue_update=on_queue_update)

    # https://fal.ai/models/fal-ai/flux/dev/image-to-image/api#schema-output
    # {'images': [
    # {'url': 'https://v3.fal.media/files/kangaroo/izqWOBYdzA7maAnW5NMOG.png',
    # 'width': 1024,
    # 'height': 768,
    # 'content_type': 'image/jpeg'}],
    # 'timings': {'inference': 2.8982661990448833},
    # 'seed': 129712570,
    # 'has_nsfw_concepts': [False],
    # 'prompt': 'A faded, old, a little burnt on the edges photo of a cat looking up at the camera and dressed as a wizard with a background of an American 1970s kitchen'}
    print(result)

def test_image_to_image_with_more_inputs():
    application_name = "fal-ai/flux/dev/image-to-image"
    image_url = "https://i.imgur.com/F9Yg1Up.png"
    prompt = (
        "A faded, a little burnt at the edges, made to look like an old photo "
        "with stains and imperfections for wear and tear photo of a beautiful "
        "blonde girl grinning happily at the camera gesturing with her hands "
        "with beautiful green eyes.")

    # https://fal.ai/models/fal-ai/flux/dev/image-to-image/api#schema-input

    # Default value is 0.95, strength of initial image.
    strength = 0.55

    # Number of inference steps to perform, default is 40.
    inference_steps = 80

    # seed

    # CFG (Classifier Free Guidance) scale is a measure of how close you want
    # model to stick to prompt when looking for related image to show you.
    # Default value is 3.5. Lower value should be more creative.
    guidance_scale = 7.5

    handler = fal_client.submit(
        application_name,
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "strength": strength,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale})

    request_id = handler.request_id
    result = fal_client.result(application_name, request_id)

    # {'images': [{'url': 'https://v3.fal.media/files/monkey/LM8C2fDhE4eG088yPLDwD.png',
    # 'width': 1072,
    # 'height': 1216,
    # 'content_type': 'image/jpeg'}],
    # 'timings': {'inference': 2.9305582970846444},
    # 'seed': 651016302,
    # 'has_nsfw_concepts': [False],
    # 'prompt': 'A faded, a little burnt at the edges, made to look like an old photo with stains and imperfections for wear and tear photo of a beautiful blonde girl grinning happily at the camera gesturing with her hands with beautiful green eyes.'}
    print(result)

def test_flux_lora_canny():
    application_name = "fal-ai/flux-lora-canny"
    image_url = "https://i.imgur.com/F9Yg1Up.png"
    prompt = (
        "A faded, a little burnt at the edges, made to look like an old photo "
        "with stains and imperfections for wear and tear photo of a beautiful "
        "blonde girl grinning happily at the camera gesturing with her hands "
        "with beautiful green eyes.")

    result = fal_client.subscribe(
        application_name,
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "num_inference_steps": 40,
            # guidance scale needs to be greater than or equal to 20
            "guidance_scale": 21.0,
            "enable_safety_checker": False},
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    print("result1", result)

    result = fal_client.subscribe(
        application_name,
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "num_inference_steps": 40,
            # guidance scale needs to be less than or equal to 40 (limit value
            # of 40)
            "guidance_scale": 39.0,
            "enable_safety_checker": False},
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    print("result2", result)

def test_flux_lora_depth():
    application_name = "fal-ai/flux-lora-depth"
    image_url = "https://i.imgur.com/TPo45xW.jpg"
    prompt = (
        "75 years later: Aerial wide shot of a dilapidated New England "
        "sprawling estate on a lake, late afternoon, with overgrown gardens "
        "on a terrace near the water after decades of decline. Seen from "
        "perspective from below from someone looking upwards from the bottom "
        "of the hill.")

    result = fal_client.subscribe(
        application_name,
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "enable_safety_checker": False,
            # Default value is 10, float.
            "guidance_scale": 2.5},
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    print("result1", result)

    result = fal_client.subscribe(
        application_name,
        arguments={
            "image_url": image_url,
            "prompt": prompt,
            "enable_safety_checker": False,
            "guidance_scale": 15.0,
            # Default value is 28
            "num_inference_steps": 40,},
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    print("result2", result)

def test_flux_pro_v1_depth():
    # Originally, for this application, you must upload to fal for a working
    # URL for the image.
    #image_url = fal_client.upload_file("/Data/Private/InputImages/Estate_001_1930s_jpg.jpg")
    #print("image_url", image_url)
    application_name = "fal-ai/flux-pro/v1/depth"
    prompt = (
        "75 years later: Aerial wide shot of a dilapidated New England "
        "sprawling estate on a lake, late afternoon, with overgrown gardens "
        "on a terrace near the water after decades of decline. Seen from "
        "perspective from below from someone looking upwards from the bottom "
        "of the hill.")
    image_url = "https://v3.fal.media/files/zebra/QJQrTXBwUv0T73VQnZkft_Estate_001_1930s_jpg.jpg"

    result = fal_client.subscribe(
        application_name,
        arguments={
            "control_image_url": image_url,
            "prompt": prompt,
            # 5 being most permissive, 1 being most strict.
            "safety_tolerance": 6,
            # Default value is 3.5, float.
            "guidance_scale": 2.2},
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    # https://fal.media/files/penguin/XRhld-0MJEqZ8C9A59gNk_4851dd6dd956466eb634d48d43da224d.jpg
    print("result1", result)

    result = fal_client.subscribe(
        application_name,
        arguments={
            "control_image_url": image_url,
            "prompt": prompt,
            "safety_tolerance": 6,
            "guidance_scale": 6.5,
            # Default value is 28
            "num_inference_steps": 40,},
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    # https://fal.media/files/penguin/AmkLe6-3oEiSNJM4C4m50_bc3ca5f4b24d4c2aa1d532599d882a57.jpg
    print("result2", result)

def test_flux_pro_v1_depth_finetuned():
    # TODO: fix, E           fal_client.client.FalClientError: Error generating image
    # TODO: E       httpx.HTTPStatusError: Server error '500 Internal Server Error' for url 'https://queue.fal.run/fal-ai/flux-pro/requests/f5e04f7e-cc8e-4ed9-90db-9f886d65a8ce'
    # E       For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500
    # fix, use a different image

    # Originally, for this application, you must upload to fal for a working
    # URL for the image.
    #image_url = fal_client.upload_file("/Data/Private/InputImages/Estate_001_1930s_jpg.jpg")
    #print("image_url", image_url)
    application_name = "fal-ai/flux-pro/v1/depth-finetuned"
    prompt = (
        "75 years later: Aerial wide shot of a dilapidated New England "
        "sprawling estate on a lake, late afternoon, with overgrown gardens "
        "on a terrace near the water after decades of decline. Seen from "
        "perspective from below from someone looking upwards from the bottom "
        "of the hill.")
    image_url = "https://v3.fal.media/files/zebra/QJQrTXBwUv0T73VQnZkft_Estate_001_1930s_jpg.jpg"

    result = fal_client.subscribe(
        application_name,
        arguments={
            "control_image_url": image_url,
            "prompt": prompt,
            # https://fal.ai/models/fal-ai/flux-pro/v1/depth-finetuned/api
            # finetune_id string required
            "finetune_id": "",
            # 5 being most permissive, 1 being most strict.
            "safety_tolerance": 6,
            # Default value is 15, float.
            "guidance_scale": 2.2,
            "finetune_strength": 0.25},
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    # https://fal.media/files/penguin/XRhld-0MJEqZ8C9A59gNk_4851dd6dd956466eb634d48d43da224d.jpg
    print("result1", result)

    result = fal_client.subscribe(
        application_name,
        arguments={
            "control_image_url": image_url,
            "prompt": prompt,
            "finetune_id": "",
            "safety_tolerance": 6,
            "guidance_scale": 25.0,
            # Default value is 28
            "num_inference_steps": 40,
            "finetune_strength": 15.0},
        with_logs=True,
        on_queue_update=on_queue_update,
    )

    # https://fal.media/files/penguin/AmkLe6-3oEiSNJM4C4m50_bc3ca5f4b24d4c2aa1d532599d882a57.jpg
    print("result2", result)

def test_flux_pro_v1_depth_with_requests():
    # Originally, for this application, you must upload to fal for a working
    # URL for the image.
    #image_url = fal_client.upload_file("/Data/Private/InputImages/Estate_001_1930s_jpg.jpg")
    #print("image_url", image_url)
    application_name = "fal-ai/flux-pro/v1/depth"
    prompt = (
        "75 years later: Aerial wide shot of a dilapidated New England "
        "sprawling estate on a lake, late afternoon, with overgrown gardens "
        "on a terrace near the water after decades of decline. Seen from "
        "perspective from below from someone looking upwards from the bottom "
        "of the hill.")
    image_url = "https://v3.fal.media/files/zebra/QJQrTXBwUv0T73VQnZkft_Estate_001_1930s_jpg.jpg"

    handler = fal_client.submit(
        application_name,
        arguments={
            "control_image_url": image_url,
            "prompt": prompt,
            # 5 being most permissive, 1 being most strict.
            "safety_tolerance": 6,
            # Default value is 3.5, float.
            "guidance_scale": 2.2},
    )

    request_id = handler.request_id

    status = fal_client.status(application_name, request_id, with_logs=True)
    print(type(status))
    print("status", status)

    result = fal_client.result(application_name, request_id)
    print(type(status))
    print("status", status)

    # https://fal.media/files/penguin/XRhld-0MJEqZ8C9A59gNk_4851dd6dd956466eb634d48d43da224d.jpg
    print("result1", result)

    print(type(status))
    print("status", status)

    # result = fal_client.submit(
    #     application_name,
    #     arguments={
    #         "control_image_url": image_url,
    #         "prompt": prompt,
    #         "safety_tolerance": 6,
    #         "guidance_scale": 6.5,
    #         # Default value is 28
    #         "num_inference_steps": 40,},
    # )

    # request_id = handler.request_id

    # status = fal_client.status(application_name, request_id, with_logs=True)
    # print("status", status)
    # # https://fal.media/files/penguin/AmkLe6-3oEiSNJM4C4m50_bc3ca5f4b24d4c2aa1d532599d882a57.jpg
    # print("result2", result)

@pytest.mark.asyncio
async def test_flux_pro_v1_depth_with_async():
    application_name = "fal-ai/flux-pro/v1/depth"
    prompt = (
        "75 years later: Aerial wide shot of a dilapidated New England "
        "sprawling estate on a lake, late afternoon, with overgrown gardens "
        "on a terrace near the water after decades of decline. Seen from "
        "perspective from below from someone looking upwards from the bottom "
        "of the hill.")
    image_url = "https://v3.fal.media/files/zebra/QJQrTXBwUv0T73VQnZkft_Estate_001_1930s_jpg.jpg"

    handler = await submit_async(
        application_name,
        arguments={
            "control_image_url": image_url,
            "prompt": prompt,
            "safety_tolerance": 6,
            "guidance_scale": 2.2
        },
    )
    
    async for event in handler.iter_events(with_logs=True):
        print(event)

    result = await handler.get()
    print("Final result:", result)
    assert result is not None
