from collections import namedtuple
from morecomputervision import (
    draw_keypoints_and_connections,
    get_maximum_sized_face,
    load_image_with_diffusers)
from morecomputervision.color_conversions import from_rgb_to_bgr

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
        det_thresh=0.5,
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

        # In insightface/python-package/insightface/app/face_analysis.py,
        # def prepare(self, ctx_id, det_thres=0.5, det_size=(640, 640)) such
        # that for each loaded model,
        # model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
        self.application.prepare(
            ctx_id=0,
            det_thresh=det_thresh,
            det_size=(det_size, det_size))

    def get_face_info_from_image(self, face_image_path):
        """
        @param height [int] typically height of original image
        @param width [int] typically width of original image
        """
        face_image = load_image_with_diffusers(face_image_path)
        face_image_cv2 = from_rgb_to_bgr(face_image)
        height, width, _ = face_image_cv2.shape

        # In insightface/python-package/insightface/app/face_analysis.py,
        # def get(self, img, max_num=0) such that max_num is used in
        # self.det_model.detect(img, max_num=max_num, metric='default') and
        # self.det_model is self.models['detection']
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
        """
        @return FaceAnalysisWrapper.PoseInformation instance
        """
        pose_image = load_image_with_diffusers(pose_image_path)
        pose_image_cv2 = from_rgb_to_bgr(pose_image)
        # In insightface/python-package/insightface/app/face_analysis.py,
        # def get(self, img, max_num=0) such that max_num is used in
        # self.det_model.detect(img, max_num=max_num, metric='default') and
        # self.det_model is self.models['detection']
        pose_info = self.application.get(pose_image_cv2)

        pose_info = pose_info[-1]
        pose_keypoints = draw_keypoints_and_connections(pose_image, pose_info['kps'])
        width, height = pose_keypoints.size

        return FaceAnalysisWrapper.PoseInformation(
            pose_keypoints = pose_keypoints,
            height = height,
            width = width)


def get_face_and_pose_info_from_images(
    model_name,
    model_root_directory,
    face_image_path,
    pose_image_path=None,
    providers=None,
    det_thresh=None,
    det_size=None):
    """
    @brief **USE THIS FUNCTION** to get face and pose information from images.

    @details By creating an object for FaceAnalysisWrapper, which wraps a
    FaceAnalysis object as a class data member, within this function, the object
    goes out of scope once the function is done. Otherwise, the FaceAnalysis
    object occupies a significant amount of VRAM from loading a model into
    memory.
    """
    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    if det_size is None:
        det_size = 256

    if det_thresh is None:
        det_thresh = 0.5

    face_analysis_wrapper = FaceAnalysisWrapper(
        name=model_name,
        root=model_root_directory,
        providers=providers,
        det_thresh=det_thresh,
        det_size=det_size)

    if pose_image_path is None:

        pose_image_path = face_image_path

    face_information = face_analysis_wrapper.get_face_info_from_image(
        face_image_path)

    pose_information = face_analysis_wrapper.get_pose_info_from_image(
        pose_image_path)

    return face_information, pose_information