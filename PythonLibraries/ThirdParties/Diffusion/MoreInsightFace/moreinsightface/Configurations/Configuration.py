from corecode.FileIO import get_project_directory_path
from pathlib import Path
import yaml

class Configuration:

    default_det_size_value = 640

    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "ThirdParties" / \
                "MoreInsightFace" / "configuration.yml"
        ):

        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        self.face_analysis_model_name = str(data["face_analysis_model_name"])
        self.face_analysis_model_directory_path = Path(data[
            "face_analysis_model_directory_path"])

        # Used by FaceAnalysis.prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640))
        # in insightface/python-package/insightface/app/face_analysis.py which
        # in turn is used by the 'detection' model for input_size.
        self.det_thresh = data["det_thresh"]
        self.det_size = data["det_size"]

        if isinstance(self.det_size, list):
            number_of_entries = len(self.det_size)
            if number_of_entries >= 2:
                self.det_size = (self.det_size[0], self.det_size[1])
            elif number_of_entries == 1:
                self.det_size = (self.det_size[0], self.det_size[0])
            else:
                self.det_size = (
                    Configuration.default_det_size_value,
                    Configuration.default_det_size_value)
        elif self.det_size != None:
            self.det_size = (self.det_size, self.det_size)
        else:
            self.det_size = (
                Configuration.default_det_size_value,
                Configuration.default_det_size_value)
