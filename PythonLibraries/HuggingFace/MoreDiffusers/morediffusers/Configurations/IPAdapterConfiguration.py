from corecode.FileIO import get_project_directory_path
from pathlib import Path
import yaml

class IPAdapterConfiguration:
    def __init__(
        self,
        configuration_path=
            get_project_directory_path() / "Configurations" / "HuggingFace" / \
                "MoreDiffusers" / "ip_adapter_configuration.yml"
        ):
        f = open(str(configuration_path), 'r')
        data = yaml.safe_load(f)
        f.close()

        self.path = data["path"]
        self.subfolder = data["subfolder"]
        self.image_filepath = data["image_filepath"]
        self.weight_names = data["weight_names"]
        self.scales = data["scales"]

        if isinstance(self.weight_names, list):

            number_of_weights = len(weight_names)

            if not isinstance(scales, list):

                self.scales = [self.scales for i in range(number_of_weights)]

            elif len(scales) < number_of_weights:

                deficient = number_of_weights - len(scales)
                for i in range(deficient):
                    self.scales.append(self.scales[-1])