from diffusers import StableDiffusionXLControlNetPipeline
import pytest

def test_StableDiffusionXLControlNetPipelineInits():

    # Requires 8 positional arguments:
    # TypeError: StableDiffusionXLControlNetPipeline.__init__() missing 8 required positional arguments: 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer', 'tokenizer_2', 'unet', 'controlnet', and 'scheduler'

    with pytest.raises(TypeError) as err:
        pipeline = StableDiffusionXLControlNetPipeline()
