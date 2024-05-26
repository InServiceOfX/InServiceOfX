from corecode.FileIO import get_project_directory_path
from pathlib import Path
import yaml

class Configuration:
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "ThirdParties" / \
                "MoreInstantID" / "configuration.yml"
        ):
        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        self.face_analysis_model_name = data["face_analysis_model_name"]
        self.face_analysis_model_directory_path = data[
            "face_analysis_model_directory_path"]
        self.control_net_model_path = data["control_net_model_path"]
        self.diffusion_model_path = data["diffusion_model_path"]
        self.ip_adapter_path = data["ip_adapter_path"]
        self.face_image_path = data["face_image_path"]
        self.pose_image_path = data["pose_image_path"]
        self.temporary_save_path = data["temporary_save_path"]
        # Used by FaceAnalysis.prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640))
        # in insightface/python-package/insightface/app/face_analysis.py which
        # in turn is used by the 'detection' model for input_size.
        self.det_size = data["det_size"]
        # IP (Image Prompt) Adapter parameters
        self.ip_adapter_image_embedding_dimension = data[
            "ip_adapter_image_embedding_dimension"]
        self.ip_adapter_number_of_tokens = data["ip_adapter_number_of_tokens"]
        self.scheduler = data["scheduler"]

    def check_if_paths_exist():
        if (not Path(self.face_analysis_model_name).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.face_analysis_model_name)
        elif (not Path(self.face_analysis_model_directory_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.face_analysis_directory_path)
        elif (not Path(self.control_net_model_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.control_net_model_path)
        elif (not Path(self.diffusion_model_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.diffusion_model_path)
        elif (not Path(self.ip_adapter_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.ip_adapter_path)
        elif (not Path(self.face_image_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.face_image_path)
        elif (not Path(self.temporary_save_path).exists()):
            raise RuntimeError(
                "Path doesn't exist: ",
                self.temporary_save_path)

        return True