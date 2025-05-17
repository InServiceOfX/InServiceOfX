from corecode.Utilities import (
    DataSubdirectories,
    )

from insightface.app import FaceAnalysis
from moreinsightface.Wrappers import (
    FaceAnalysisWrapper,
    get_face_and_pose_info_from_images
    )

import insightface

import pytest
import torch

data_sub_dirs = DataSubdirectories()

def test_FaceAnalysisWrapper_inits():

    face_analysis = FaceAnalysisWrapper(
        "antelopev2",
        data_sub_dirs.ModelsDiffusion / "InstantX",
        det_size=(640, 640)
        )

    assert isinstance(face_analysis.application, FaceAnalysis)

def test_FaceAnalysisWrapper_gets_face_embedding():
    """
    From
    https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#face-model 
    given the example "To use IP-Adapter FaceID models, ..."
    """

    face_analysis = FaceAnalysisWrapper(
        "antelopev2",
        data_sub_dirs.ModelsDiffusion / "InstantX",
        det_size=(640, 640)
        )

    website_address = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png"

    results = face_analysis.get_face_embedding_from_image(website_address)

    assert isinstance(results.face_info, list)
    assert len(results.face_info) == 1
    assert isinstance(results.face_info[0], insightface.app.common.Face)

    assert isinstance(results.image, torch.Tensor)
    assert isinstance(results.ref_images_embeds, torch.Tensor)
    assert isinstance(results.neg_ref_images_embeds, torch.Tensor)
    assert isinstance(results.id_embeds, torch.Tensor)

    assert results.image.shape == torch.Size([512])
    assert results.ref_images_embeds.shape == torch.Size([1, 1, 1, 512])
    assert results.neg_ref_images_embeds.shape == torch.Size([1, 1, 1, 512])
    assert results.id_embeds.shape == torch.Size([2, 1, 1, 512])

    assert results.image.size() == torch.Size([512])
    assert results.image.dim() == 1

    assert results.ref_images_embeds.size() == torch.Size([1, 1, 1, 512])
    assert results.ref_images_embeds.dim() == 4

    assert results.neg_ref_images_embeds.size() == torch.Size([1, 1, 1, 512])
    assert results.neg_ref_images_embeds.dim() == 4

    assert results.id_embeds.size() == torch.Size([2, 1, 1, 512])
    assert results.id_embeds.dim() == 4


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
