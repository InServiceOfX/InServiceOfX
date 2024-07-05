from morecomputervision import load_image_with_diffusers

from PIL import Image

def test_load_image_with_diffusers_can_load_from_website():
	"""
	From
	https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#face-model	
	given the example "To use IP-Adapter FaceID models, ..."
	"""
	website_address = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png"

	image = load_image_with_diffusers(website_address)

	assert isinstance(image, Image.Image)
	assert image.size == (1024, 1024)