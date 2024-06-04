import cv2
import numpy as np

def from_rgb_to_bgr(image):
	if type(image) == np.ndarray:
		return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	else:
		return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)