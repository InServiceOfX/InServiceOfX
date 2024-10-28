from corecode.Conversions.Images import encode_image_to_base64

from pathlib import Path

test_data_directory = Path(__file__).parents[5] / "ThirdParties" / \
    "MoreInsightFace" / "tests" / "TestData" / "Images"

def test_encode_image_to_base64_encodes():
    lenna_image_path = test_data_directory / "Lenna_(test_image).png"
    encoded_image = encode_image_to_base64(lenna_image_path)
    assert encoded_image is not None
    assert isinstance(encoded_image, str)
    
    assert len(encoded_image) == 631776
    
    assert encoded_image.startswith("iVBORw0KGgo")
    
    # Check some specific positions in the middle
    assert encoded_image[1000:1010] == "9SvHO3V3Oy"
    
    # Check the end of the string
    assert encoded_image[-10:] == "VORK5CYII="

    image_path = test_data_directory / "23BM9O8mTKs.jpg"
    encoded_image = encode_image_to_base64(image_path)
    assert encoded_image is not None
    assert isinstance(encoded_image, str)
    assert len(encoded_image) == 1278896
    assert encoded_image.startswith("/9j/4AAQSkZJRgA")
    assert encoded_image[1000:1010] == "Pqj9si7+En"
    assert encoded_image[-10:] == "F/8E//2Q=="
