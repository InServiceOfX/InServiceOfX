from corecode.Utilities import load_environment_file
from pathlib import Path

from morelumaai.Wrappers import CreateImage

load_environment_file()

def test_CreateImage_init():
    create_image = CreateImage()

def test_CreateImage_generate():
    create_image = CreateImage()
    response = create_image.generate(
        "Zoom in towards nexus or center slightly",
        image_ref=CreateImage.create_image_ref(
            "https://www.michaeldivine.com/wp-content/uploads/2021/01/Station-to-Station-1.jpg",
            weight=1.9
        )
    )

    print(response)

def test_CreateImage_generate_large_weight():
    create_image = CreateImage()
    response = create_image.generate(
        "Zoom in towards nexus or center slightly",
        image_ref=CreateImage.create_image_ref(
            "https://www.michaeldivine.com/wp-content/uploads/2021/01/Station-to-Station-1.jpg",
            weight=11.9
        )
    )

    print(response)

def test_CreateImage_generate_larger_weight():
    create_image = CreateImage()
    response = create_image.generate(
        "Zoom in towards nexus or center slightly and subtly without much change to image",
        image_ref=CreateImage.create_image_ref(
            "https://www.michaeldivine.com/wp-content/uploads/2021/01/Station-to-Station-1.jpg",
            weight=111.9
        )
    )

    print(response)

def test_CreateImage_generate_with_modify_image():
    create_image = CreateImage()
    response = create_image.generate(
        model="photon-flash-1",
        prompt="Zoom in towards nexus or center slightly",
        modify_image_ref=CreateImage.create_modify_image_ref(
            "https://www.michaeldivine.com/wp-content/uploads/2021/01/Station-to-Station-1.jpg",
            # weight=11.9 resulted in state='failed' in response
            weight=1.0
        )
    )

    print(response)