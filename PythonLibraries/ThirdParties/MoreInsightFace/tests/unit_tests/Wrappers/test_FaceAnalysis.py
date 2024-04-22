from moreinsightface.Wrappers import FaceAnalysisWrapper
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

test_data_directory = Path(__file__).resolve().parent.parent.parent / "TestData"

def test_FaceAnalysisWrapper_inits():
	test_directory_path = test_data_directory / "Models" / "Diffusion" / "InstantX"
	assert test_directory_path.exists()

	app = FaceAnalysisWrapper("buffalo_sc", test_directory_path)

	assert True

def test_get_face_info_from_image_gets():
	test_directory_path = test_data_directory / "Models" / "Diffusion" / "InstantX"

	image_file_path = test_data_directory / "Images" / "Lenna_(test_image).png"
	assert image_file_path.exists()

	app = FaceAnalysisWrapper("buffalo_sc", test_directory_path)

	face_information = app.get_face_info_from_image(image_file_path)
	assert type(face_information.face_embedding) == np.ndarray
	assert type(face_information.face_keypoints) == Image.Image
	assert face_information.height == 512
	assert face_information.width == 512

def test_get_face_info_from_image_gets():
	test_directory_path = test_data_directory / "Models" / "Diffusion" / "InstantX"

	image_file_path = test_data_directory / "Images" / "23BM9O8mTKs.jpg"
	assert image_file_path.exists()

	app = FaceAnalysisWrapper("buffalo_sc", test_directory_path)

	pose_information = app.get_pose_info_from_image(image_file_path)

	assert type(pose_information.pose_keypoints) == Image.Image
	assert type(pose_information.height) == 1280
	assert type(pose_information.width) == 1281