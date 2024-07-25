from diffusers import MotionAdapter

def create_motion_adapter(
	model_subdirectory,
    torch_dtype=None,
    use_safetensors=None,
    ):

    if torch_dtype==None:
        # pipelines/i2vgen_xl/pipeline_i2vgen_xl.py
        # implements I2VGenXLPipeline
        # from_pretrained(..) defined in DiffusionPipeline in
        # diffusers/src/diffusers/pipelines/pipeline_utils.py
        pipe = MotionAdapter.from_pretrained(
            str(model_subdirectory),
            local_files_only=True,
            use_safetensors=use_safetensors)
    else:
        pipe = MotionAdapter.from_pretrained(
            str(model_subdirectory),
            torch_dtype=torch_dtype,
            local_files_only=True,
            use_safetensors=use_safetensors)

    return pipe