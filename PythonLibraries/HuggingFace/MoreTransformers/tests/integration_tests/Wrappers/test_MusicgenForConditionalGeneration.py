from corecode.Utilities import DataSubdirectories

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "musicgen-medium"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-medium"

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

def test_pipeline_constructs():
    synthesizer = pipeline("text-to-audio", pretrained_model_name_or_path)

    # This warning is obtained:
    # `torch.nn.functional.scaled_dot_product_attention` does not support having
    # an empty attention mask. Falling back to the manual attention
    # implementation. This warning can be removed using the argument
    # `attn_implementation="eager"` when loading the model.Note that this
    # probably happens because `guidance_scale>1` or because you used 
    # `get_unconditional_inputs`. See
    # https://github.com/huggingface/transformers/issues/31189 for more
    # information.
    music = synthesizer(
        "lo-fi music with a soothing melody",
        forward_params={"do_sample": True})

    scipy.io.wavfile.write(
        "musicgen_out.wav",
        rate=music["sampling_rate"],
        data=music["audio"])