from collections import namedtuple
from corecode.ComputerVision import (
    draw_keypoints_and_connections,
    get_maximum_sized_face)
from corecode.ComputerVision.color_conversions import from_rgb_to_bgr
from corecode.FileIO import load_image_with_diffusers

from insightface.app import FaceAnalysis

import cv2
from pathlib import Path

class FaceAnalysisWrapper:
    FaceInformation = namedtuple(
        "FaceInformation",
        ["face_embedding", "face_keypoints", "height", "width"])

    PoseInformation = namedtuple(
        "PoseInformation",
        ["pose_keypoints", "height", "width"])

    def __init__(
        self,
        name,
        root,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        det_size=256
        ):
        """
        @param name Name of the model; it's expected this is also the name of
        the subdirectory that has root as its "parent" directory and that this
        subdirectory would contain the model.
        e.g. name="buffalo_l"

        @param root Path of the directory containing the subdirectory called
        name, which would then contain the model files.
        """
        self.application = FaceAnalysis(
            name=name,
            root=root,
            providers=providers)

        self.application.prepare(ctx_id=0, det_size=(det_size, det_size))

    def get_face_info_from_image(self, face_image_path):
        face_image = load_image_with_diffusers(face_image_path)
        face_image_cv2 = from_rgb_to_bgr(face_image)
        height, width, _ = face_image_cv2.shape
        face_info = self.application.get(face_image_cv2)

        # Get the maximum sized face.
        face_info = get_maximum_sized_face(face_info)

        face_keypoints = draw_keypoints_and_connections(face_image, face_info['kps'])

        return FaceAnalysisWrapper.FaceInformation(
            face_embedding=face_info["embedding"],
            face_keypoints=face_keypoints,
            height=height,
            width=width)

    def get_pose_info_from_image(self, pose_image_path):
        pose_image = load_image_with_diffusers(pose_image_path)
        pose_image_cv2 = from_rgb_to_bgr(pose_image)
        pose_info = self.application.get(pose_image_cv2)

        pose_info = pose_info[-1]
        pose_keypoints = draw_keypoints_and_connections(pose_image, pose_info['kps'])
        width, height = pose_keypoints.size

        return FaceAnalysisWrapper.PoseInformation(
            pose_keypoints = pose_keypoints,
            height = height,
            width = width)
