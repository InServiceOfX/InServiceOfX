from PIL import Image

import cv2
import math
import numpy as np

def draw_keypoints_and_connections(image_pil, keypoints, color_list=None):
	if color_list is None:
		color_list = [
			(255, 0, 0),
			(0, 255, 0),
			(0, 0, 255),
			(255, 255, 0),
			(255, 0, 255)]

	stickwidth = 4
	limb_sequence = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
	keypoints = np.array(keypoints)

	width, height = image_pil.size
	output_image = np.zeros([height, width, 3])

	for limb in limb_sequence:
		color = color_list[limb[0]]
		x_coordinates = keypoints[limb][:, 0]
		y_coordinates = keypoints[limb][:, 1]
		length = np.hypot(
			x_coordinates[0] - x_coordinates[1],
			y_coordinates[0] - y_coordinates[1])
		angle = math.degrees(
			math.atan2(
				y_coordinates[0] - y_coordinates[1],
				x_coordinates[0] - x_coordinates[1]))
		polygon = cv2.ellipse2Poly(
			(int(np.mean(x_coordinates)), int(np.mean(y_coordinates))),
			(int(length / 2), stickwidth),
			int(angle),
			0,
			360,
			1)
		output_image = cv2.fillConvexPoly(output_image.copy(), polygon, color)
	output_image = (output_image * 0.6).astype(np.uint8)

	for index, keypoint in enumerate(keypoints):
		color = color_list[index % len(color_list)]
		x, y = keypoint
		output_image = cv2.circle(
			output_image.copy(),
			(int(x), int(y)),
			10,
			color,
			-1)

	output_image_pil = Image.fromarray(output_image.astype(np.uint8))
	return output_image_pil