from corecode.Utilities import (
    get_user_input,
    FloatParameter,
    IntParameter,
    StringParameter)

from pathlib import Path


class UserInput:

    def __init__(self, configuration,):

        self.system_prompt = StringParameter(
            get_user_input(str, "Prompt for system: ", ""))
        self.user_prompt = StringParameter(
            get_user_input(str, "Prompt User prompt: ", ""))

        print("system_prompt: ", self.system_prompt.value)
        print("user_prompt: ", self.user_prompt.value)

        self.model_name = Path(configuration.model_path).name

    def create_messages(self):
        return [
            {"role": "system", "content": self.system_prompt.value},
            {"role": "user", "content": self.user_prompt.value}]
    
