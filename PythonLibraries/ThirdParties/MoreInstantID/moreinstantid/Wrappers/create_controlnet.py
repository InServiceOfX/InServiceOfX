from diffusers.models import ControlNetModel

def create_controlnet(controlnet_path, torch_dtype=None):
    """
    @details Look at diffusers/src/diffusers/models/controlnet.py,
    ControlNetModel inherits from ModelMixin. See
    diffusers/src/diffusers/models/modeling_utils.py for
    ModelMixin.from_pretrained and all possible key word arguments (kwargs)
    """
    if torch_dtype==None:
        return ControlNetModel.from_pretrained(
            str(controlnet_path),
            local_files_only=True)
    else:
        return ControlNetModel.from_pretrained(
            str(controlnet_path),
            local_files_only=True,
            torch_dtype=torch_dtype)