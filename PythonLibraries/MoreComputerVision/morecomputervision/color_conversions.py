import cv2
import numpy as np

def from_rgb_to_bgr(image):
	if type(image) == np.ndarray:
		return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	else:
		return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def from_bgr_to_rgb(image):
	"""
	From
	https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#face-model	
	given the example "To use IP-Adapter FaceID models, ..."
	"""
	return cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)