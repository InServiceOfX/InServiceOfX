from corecode.Utilities import (
    load_environment_file,
    get_environment_variable
)
from pathlib import Path

from morelumaai.Configuration.APIDirectlyConfigurations import Ray2Configuration
from morelumaai.Wrappers.APIDirectly import Ray2ImageToVideo

import requests

load_environment_file()

def test_Ray2ImageToVideo_init():
    configuration = Ray2Configuration()
    ray2_image_to_video = Ray2ImageToVideo()
    payload = ray2_image_to_video.create_payload(
        "A beautiful woman",
        "https://www.michaeldivine.com/wp-content/uploads/2021/01/Praise-the-Oontsa-Oontsa-3.jpg",
    )
    assert payload["prompt"] == "A beautiful woman"
    assert payload["keyframes"]["frame0"]["type"] == "image"
    assert payload["keyframes"]["frame0"]["url"] == \
        "https://www.michaeldivine.com/wp-content/uploads/2021/01/Praise-the-Oontsa-Oontsa-3.jpg"

    headers = ray2_image_to_video._create_headers("luma_fake_api_key")
    assert headers["accept"] == "application/json"
    assert headers["authorization"] == "Bearer luma_fake_api_key"
    assert headers["content-type"] == "application/json"

def test_Ray2ImageToVideo_generate():
    configuration = Ray2Configuration()
    ray2_image_to_video = Ray2ImageToVideo()
    response = ray2_image_to_video.generate(
        get_environment_variable("LUMAAI_API_KEY"),
        "Seated figure",
        #"https://www.michaeldivine.com/wp-content/uploads/2021/01/Praise-the-Oontsa-Oontsa-3.jpg",
        "https://www.michaeldivine.com/wp-content/uploads/2025/03/Samadhi50mb-scaled.jpg"
    )
    print(response)
