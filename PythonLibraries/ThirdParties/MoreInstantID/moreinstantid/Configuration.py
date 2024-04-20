from corecode.FileIO import get_project_directory_path

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
		self.face_analysis_directory_path = data[
			"face_analysis_model_directory_path"]
		self.control_net_model_path = data["control_net_model_path"]