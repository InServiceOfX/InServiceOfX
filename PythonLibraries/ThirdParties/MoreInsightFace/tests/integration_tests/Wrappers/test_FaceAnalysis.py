from corecode.Utilities import (
    DataSubdirectories,
    )
from moreinsightface.Wrappers import get_face_and_pose_info_from_images

import pytest

data_sub_dirs = DataSubdirectories()

def test_get_face_and_pose_info_from_images_works_with_default_values():

    default_result = get_face_and_pose_info_from_images(
        "antelopev2",
        data_sub_dirs.ModelsDiffusion / "InstantX",
        data_sub_dirs.Public / "Images" / "Playboy" / \
            "LennaSjööblom_LenaForsén" / "qShFGgyfM2o8OTptV6bGyf_9QVDc7x2ua38DJyWci9nsRi8u2ZJunn27MMP8vji6wZmUna5cR7TDxpC_p1wrzVinINlkVsl8tB6JY3RF81L9bAU38H4N-EQwDtEWWUT2.jpeg",
        data_sub_dirs.Public / "Images" / "Playboy" / "TeddiSmith" / \
            "tumblr_oythctEdBF1wykvxvo1_1280.jpg")

    assert default_result[0].face_embedding.shape[0] == 512
    assert default_result[0].height == 699
    assert default_result[0].width == 472
    assert default_result[1].height == 800
    assert default_result[1].width == 773

def test_get_face_and_pose_info_from_images_different_for_larger_det_size():

    default_result = get_face_and_pose_info_from_images(
        "antelopev2",
        data_sub_dirs.ModelsDiffusion / "InstantX",
        data_sub_dirs.Public / "Images" / "Playboy" / \
            "LennaSjööblom_LenaForsén" / "qShFGgyfM2o8OTptV6bGyf_9QVDc7x2ua38DJyWci9nsRi8u2ZJunn27MMP8vji6wZmUna5cR7TDxpC_p1wrzVinINlkVsl8tB6JY3RF81L9bAU38H4N-EQwDtEWWUT2.jpeg",
        data_sub_dirs.Public / "Images" / "Playboy" / "TeddiSmith" / \
            "tumblr_oythctEdBF1wykvxvo1_1280.jpg")

    original_default_det_result = get_face_and_pose_info_from_images(
        "antelopev2",
        data_sub_dirs.ModelsDiffusion / "InstantX",
        data_sub_dirs.Public / "Images" / "Playboy" / \
            "LennaSjööblom_LenaForsén" / "qShFGgyfM2o8OTptV6bGyf_9QVDc7x2ua38DJyWci9nsRi8u2ZJunn27MMP8vji6wZmUna5cR7TDxpC_p1wrzVinINlkVsl8tB6JY3RF81L9bAU38H4N-EQwDtEWWUT2.jpeg",
        data_sub_dirs.Public / "Images" / "Playboy" / "TeddiSmith" / \
            "tumblr_oythctEdBF1wykvxvo1_1280.jpg",
        det_size=640)
    
    assert default_result[0].face_embedding.all() == \
        original_default_det_result[0].face_embedding.all()
    assert default_result[0].face_keypoints != \
        original_default_det_result[0].face_keypoints
    assert default_result[0].height == original_default_det_result[0].height
    assert default_result[0].width == original_default_det_result[0].width
    assert default_result[1].pose_keypoints != \
        original_default_det_result[1].pose_keypoints
    assert default_result[1].height == original_default_det_result[1].height
    assert default_result[1].width == original_default_det_result[1].width
