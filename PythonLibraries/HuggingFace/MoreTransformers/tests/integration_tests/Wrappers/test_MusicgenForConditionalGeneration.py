from corecode.Utilities import DataSubdirectories
from corecode.Utilities import clear_torch_cache_and_collect_garbage

data_sub_dirs = DataSubdirectories()

pretrained_model_name_or_path = \
    data_sub_dirs.Models / "Generation" / "facebook" / "musicgen-medium"
if not pretrained_model_name_or_path.exists():
    pretrained_model_name_or_path = \
        data_sub_dirs.Data.parent / "Data1" / "Models" / "Generation" / \
            "facebook" / "musicgen-medium"

from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers.models.encodec import EncodecFeatureExtractor
from transformers.models.musicgen import MusicgenProcessor
from transformers.models.t5 import (T5EncoderModel, T5TokenizerFast)
import scipy

def test_AutoProcessor_from_pretrained():
    """
    class AutoProcessor found in transformers/models/auto/processing_auto.py

    class MusicgenProcessor(ProcessorMixin) found in
    transformers/models/musicgen/processing_musicgen.py
    """
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)

    assert isinstance(processor, MusicgenProcessor)
    assert not processor._in_target_context_manager
    assert isinstance(processor.feature_extractor, EncodecFeatureExtractor)
    assert isinstance(processor.tokenizer, T5TokenizerFast)

def test_MusicgenForConditionalGeneration_from_pretrained():
    """
    class MusicgenForConditionalGeneration(PreTrainedModel, GenerationMixin)
    found in transformers/models/musicgen/modeling_musicgen.py
    """
    model = MusicgenForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path)

    assert isinstance(model, MusicgenForConditionalGeneration)
    assert isinstance(model.text_encoder, T5EncoderModel)

def test_run_inference_via_transformers_modelling():
    """
    https://huggingface.co/facebook/musicgen-medium
    """
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
    model = MusicgenForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path)

    inputs = processor(
        text=[
            "80s pop track with bassy drums and synth",
            "90s rock song with loud guitars and heavy drums"
        ],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs, max_new_tokens=256)

    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(
        "musicgen_out.wav",
        rate=sampling_rate,
        data=audio_values[0, 0].numpy())

def test_use_MusicgenProcessor():
    """
    In preprocessor_config.json, of the model repository,
    "processor_class": "MusicgenProcessor",

    In src/transformers/models/musicgen/processing_musicgen.py,
    class MusicgenProcessor(ProcessorMixin)

    In src/transformers/processing_utils.py,
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        ...
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,):
    ...
    processor_dict, kwargs = cls.get_processor_dict(
        pretrained_model_name_or_path,
        **kwargs)
    return cls.from_args_and_dict(args, processor_dict, **kwargs)

    In src/transformers/processing_utils.py,
    @classmethod
    def from_args_and_dict(..):
        ...
        processor = cls(*args, **processor_dict)
        ...
        return processor
    """
    processor = MusicgenProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=True,
        device_map="cuda:0")

    assert isinstance(processor, MusicgenProcessor)

    inputs = processor(
        text="80s pop track with bassy drums and synth",
        padding=True,
        return_tensors="pt",)

    del processor

    # Without this line, you get this error:
    # RuntimeError:
    # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper_CUDA__index_select)
    inputs.to("cuda:0")

    # See src/transformers/modeling_utils.py for
    # class PreTrainedModel(nn.Module, ..)
    # @classmethod
    # def from_pretrained(..)
    model = MusicgenForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=True,
        attn_implementation="eager",
        # ValueError: T5EncoderModel does not support Flash Attention 2.0 yet. Please request to add support where the model is hosted, on its model hub page: https://huggingface.co/t5-base/discussions/new or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new
        #attn_implementation="flash_attention_2",
        # torch.float16, torch.bfloat16, or ftorch.float
        # If not specified, the model will get loaded in torch.float (fp32).
        # "auto" A torch_dtype entry in config.json file of model will be
        # attempted.
        #torch_dtype=torch.float,
        device_map="cuda:0")

    # From class MusicgenForConditionalGeneration(..),
    # from code comments for
    # def get_unconditional_inputs(self, num_samples=1)
    # max_new_tokens
    # Number of tokens to generate for each sample. More tokens means longer
    # audio samples, at the expense of longer inference (since more audio tokens
    # need to be generated per sample).
    audio_values = model.generate(**inputs, max_new_tokens=256)

    # Move tensor to CPU before converting to numpy
    audio_values_cpu = audio_values.cpu()
    
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(
        "musicgen_out1.wav",
        rate=sampling_rate,
        data=audio_values_cpu[0, 0].numpy())

def test_consider_GenerationConfig():
    """
    Notice that in the repository from facebook, musicgen-medium, there is a
    generation_config.json file with key-value pairs for
    "do_sample": true,
    "guidance_scale": 3.0,
    "max_length": 1500

    On the other hand, in src/transformers/generation/configuration_utils.py,
    look at class GenerationConfig(..) as generate(..) code takes kwargs and
    inputs them into a GenerationConfig.
    """

    processor = MusicgenProcessor.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=True,
        device_map="cuda:0")

    inputs = processor(
        text="80s pop track with bassy drums and synth and beach like guitar",
        padding=True,
        return_tensors="pt",)

    del processor

    inputs.to("cuda:0")

    model = MusicgenForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        local_files_only=True,
        attn_implementation="eager",
        device_map="cuda:0")

    audio_values = model.generate(
        **inputs,
        max_new_tokens=512,
        # defaults to 1.0
        # value used to module the next token probabilities.
        temperature=1.0,
        # defaults to 50 if not set
        # number of highest probability vocabulary tokens to keep for
        # top-k-filtering.
        top_k=50,
        # defaults to 1.0
        # If set to float < 1, only smallest set of most probable tokens with
        # probabilities that add up to top_p or higher are kept for generation. 
        top_p=1.0)

    # Move tensor to CPU before converting to numpy
    audio_values_cpu = audio_values.cpu()
    
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(
        "musicgen_out2.wav",
        rate=sampling_rate,
        data=audio_values_cpu[0, 0].numpy())