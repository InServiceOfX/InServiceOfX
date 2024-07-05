from collections import namedtuple
from morecomputervision import (
    draw_keypoints_and_connections,
    get_maximum_sized_face,
    load_image_with_diffusers)
from morecomputervision.color_conversions import (
    from_rgb_to_bgr,
    from_bgr_to_rgb)

from insightface.app import FaceAnalysis

import cv2
from collections import namedtuple
import torch
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

        if isinstance(det_size, int):
            self.application.prepare(
                ctx_id=0,
                det_thresh=det_thresh,
                det_size=(det_size, det_size))
        elif det_size != None:
            self.application.prepare(
                ctx_id=0,
                det_thresh=det_thresh,
                det_size=det_size)
        else:
            raise ValueError(
                f"Expected positive integer or tuple, got NoneType")

    def get_face_embedding_from_image(self, face_image_path):
        """
        From
        https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#face-model 
        given the example "To use IP-Adapter FaceID models, ..."

        @return id_embeds torch.Tensor
        id_embeds did not have .to(dtype=torch.float16, device="cuda") run on it
        yet as a torch.Tensor instance, as was done in
        https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#face-model
        for the example "To use IP-Adapter FaceID models, first extract face
        embeddings with insightface."
        """
        face_image = load_image_with_diffusers(face_image_path)
        face_image_cv2 = from_bgr_to_rgb(face_image)
        face_info = self.application.get(face_image_cv2)

        image = torch.from_numpy(face_info[0].normed_embedding)

        ref_images_embeds = []
        ref_images_embeds.append(image.unsqueeze(0))
        ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)

        neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
        id_embeds = torch.cat([neg_ref_images_embeds, ref_images_embeds])

        # Look at the example "To use IP-Adapter FaceID models, first extract face embeddings"
        # in
        # https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#face-model
        # and observe that we hadn't run
        # .to(dtype=torch.float16, device="cuda")
        # on id_embeds. Leave it up to the user to decide what to do.

        FaceEmbeddingResult = namedtuple(
            'FaceEmbeddingResult',
            [
                'face_info',
                'image',
                'ref_images_embeds',
                'neg_ref_images_embeds',
                'id_embeds'])
        return FaceEmbeddingResult(
            face_info=face_info,
            image=image,
            ref_images_embeds=ref_images_embeds,
            neg_ref_images_embeds=neg_ref_images_embeds,
            id_embeds=id_embeds)


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