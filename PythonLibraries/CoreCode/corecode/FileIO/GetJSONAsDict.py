import json

class GetJSONAsDict:
    # This is hardcoded as the configuration file to look for in Hugging Face's
    # diffusers library.
    diffuser_config_name = "model_index.json"

    @staticmethod
    def get_json_file_as_dict(filepath):
        f = open(filepath)
        json_dict = json.load(f)
        f.close()
        return json_dict

    @staticmethod
    def get_diffuser_config_as_dict(model_path, author=None, model_name=None):
        """
        @param author, default None. e.g. author="runwayml"
        @param model_name, default None, e.g. model_name="stable-diffusion-v1.5"
        """
        if author == None or model_name==None:
            return GetJSONAsDict.get_json_file_as_dict(
                model_path / GetJSONAsDict.diffuser_config_name)
        else:
            filepath = model_path / author / model_name / \
                GetJSONAsDict.diffuser_config_name
            return GetJSONAsDict.get_json_file_as_dict(filepath)
