from diffusers.models import ControlNetModel

def create_controlnet(controlnet_path, torch_dtype=None):
	if torch_dtype==None:
		return ControlNetModel.from_pretrained(str(controlnet_path))
	else:
		return ControlNetModel.from_pretrained(
			str(controlnet_path),
			torch_dtype=torch_dtype)