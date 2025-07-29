from corecode.Utilities import (
    DataSubdirectories,
    )
from moretransformers.Configurations import GenerationConfiguration
from pathlib import Path
from transformers import (
    AutoProcessor,
    DiaProcessor,
    DiaForConditionalGeneration,
    )

import io
import pytest
import torchaudio

data_sub_dirs = DataSubdirectories()

relative_model_path = \
    "Models/Generative/TextToSpeech/nari-labs/Dia-1.6B-0626"

is_model_downloaded = False
model_path = None

for path in data_sub_dirs.DataPaths:
    path = Path(path)
    if (path / relative_model_path).exists():
        is_model_downloaded = True
        model_path = path / relative_model_path
        break

model_not_downloaded_message = f"Model not downloaded: {relative_model_path}"

text = [
    (
        "[S1] Dia is an open weights text to dialogue model. [S2] You get full "
        "control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try "
        "it now on Git hub or Hugging Face."
    )
]

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_AutoProcessor_works():
    """
    https://github.com/nari-labs/dia/blob/main/hf.py
    """
    processor = AutoProcessor.from_pretrained(model_path)
    assert isinstance(processor, DiaProcessor)

    # See processing_dia.py, class DiaFeatureExtractor(SequenceFeatureExtractor)
    # def __call__(..) for call options.
    inputs = processor(text, padding=True, return_tensors="pt").to("cuda:0")

    model = DiaForConditionalGeneration.from_pretrained(model_path).to("cuda:0")

    outputs = model.generate(
        **inputs,
        max_new_tokens=3072,
        guidance_scale=3.0,
        temperature=1.8,
        top_p=0.90,
        top_k=45,
        )

    outputs = processor.batch_decode(outputs)
    assert isinstance(outputs, list)
    processor.save_audio(outputs, "test_Dia.mp3")

text_1 = [
    (
        "[S1] Justin Timberlake's concert performances in 2025 have been all "
        "over my For You pages on Tik Tok. The crowd sings more than he does."
        "[S2] I'm watching the Tik Toks too and you could've just emailed the "
        "effort. [S1] And Justin looks drunk or on something in the media "
        "photos. [S2] Jessica Biel is not impressed (sarcasm)."
    )
]

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_DiaProcessor_works():
    """
    See transformers/processing_utils.py, class ProcessorMixin,
    def from_pretrained(..)

    See feature_extraction_dia for more options on sampling_rate, feature_size,
    etc.

    See tokenization_dia.py for more options in def __init__(..) which can
    configured by kwargs when using from_pretrained(..), such as max_length,
    defaults to 1024, max length of sequences when encoding.
    """
    processor = DiaProcessor.from_pretrained(
        model_path,
        # Obtain this error:
        # E                   OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
        # E                   Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
        # /ThirdParty/transformers/src/transformers/utils/hub.py:550: OSError
        #local_files_only=True,)
        )

    # inputs is of type
    # <class 'transformers.feature_extraction_utils.BatchFeature'>
    inputs = processor(text_1, padding=True, return_tensors="pt").to("cuda:0")
    model = DiaForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True).to("cuda:0")

    generation_config = GenerationConfiguration(
        max_new_tokens=3072,
        temperature=1.8,
        top_p=0.90,
        top_k=45,
        )

    kwargs = generation_config.to_dict()
    kwargs["guidance_scale"] = 3.0

    # This works, but model.generate(**inputs, **kwargs) doesn't; I get this
    # error:
    # The following generation flags are not valid and may be ignored:
    # ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for
    # more details.
    outputs = model.generate(
        **inputs,
        max_new_tokens=kwargs["max_new_tokens"],
        guidance_scale=kwargs["guidance_scale"],
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"]
        )

    outputs = processor.batch_decode(outputs)
    processor.save_audio(outputs, "test_Dia_1.mp3")

from datasets import load_dataset, Audio

@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_generation_with_text_and_audio_voice_cloning():
    """
    See 
    https://huggingface.co/nari-labs/Dia-1.6B-0626#generation-with-text-and-audio-voice-cloning
    """
    ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
    # <class 'datasets.arrow_dataset.Dataset'>
    # print(type(ds))

    # ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    # # 12
    # print(len(ds))
    # # Obtained this error for the following line of code:
    # # NotImplementedError: Could not run 'torchcodec_ns::add_audio_stream' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build)
    # #audio = ds[-1]["audio"]["array"]
    # # ['conversation_id', 'turn_id', 'speaker_id', 'text', 'audio']
    # print(ds.column_names)
    # # (12, 5)
    # print(ds.shape)
    # # {'conversation_id': Value('int64'), 'turn_id': Value('int64'), 'speaker_id': Value('int64'), 'text': Value('string'), 'audio': Audio(sampling_rate=16000, decode=True, stream_index=None)}
    # print(ds.features)

    ds_audio_only = ds.select_columns(["audio"])
    # <class 'datasets.arrow_dataset.Dataset'>
    #print(type(ds_audio_only))

    audio_columns = ds_audio_only.data.column("audio")
    # <class 'pyarrow.lib.ChunkedArray'>
    #print(type(audio_columns))
    audio_data = audio_columns[-1]
    # <class 'pyarrow.lib.StructScalar'>
    #print(type(audio_data))

    audio_dict = audio_data.as_py()
    # class 'dict'>
    #print(type(audio_dict))
    # dict_keys(['bytes', 'path'])
    #print(audio_dict.keys())
    # This is None
    #print(audio_dict["path"])

    audio_bytes = audio_dict["bytes"]
    # <class 'bytes'>
    #print(type(audio_bytes))

    # Fails:
    # /usr/local/lib/python3.12/dist-packages/datasets/features/audio.py:198: in decode_example
    #     audio = AudioDecoder(bytes, stream_index=self.stream_index, sample_rate=self.sampling_rate)
    # /usr/local/lib/python3.12/dist-packages/torchcodec/decoders/_audio_decoder.py:64: in __init__
    #     core.add_audio_stream(
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    # self = <OpOverload(op='torchcodec_ns.add_audio_stream', overload='default')>
    # args = (tensor([                  1,           941536864,           941537216,
    #                   941537216,                   1, 4613832359700398831,
    #                           1, 4690342726636404736]),)
    # kwargs = {'num_channels': None, 'sample_rate': 16000, 'stream_index': None}

    #     def __call__(self, /, *args, **kwargs):
    # >       return self._op(*args, **kwargs)
    # E       NotImplementedError: Could not run 'torchcodec_ns::add_audio_stream' with arguments from the 'CPU' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'torchcodec_ns::add_audio_stream' is only available for these backends: [CUDA, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMTIA, AutogradMeta, Tracer, AutocastCPU, AutocastXPU, AutocastMPS, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].
    # E       
    # E       CUDA: registered at /__w/torchcodec/torchcodec/pytorch/torchcodec/src/torchcodec/_core/custom_ops.cpp:702 [kernel]
    # E       Meta: registered at /dev/null:198 [kernel]
    # E       BackendSelect: fallthrough registered at /opt/pytorch/pytorch/aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
    # E       Python: registered at /opt/pytorch/pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:194 [backend fallback]
    # E       FuncTorchDynamicLayerBackMode: registered at /opt/pytorch/pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:503 [backend fallback]
    # E       Functionalize: registered at /opt/pytorch/pytorch/aten/src/ATen/FunctionalizeFallbackKernel.cpp:349 [backend fallback]
    # E       Named: registered at /opt/pytorch/pytorch/aten/src/ATen/core/NamedRegistrations.cpp:7 [backend fallback]
    # E       Conjugate: registered at /opt/pytorch/pytorch/aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
    # E       Negative: registered at /opt/pytorch/pytorch/aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
    # E       ZeroTensor: registered at /opt/pytorch/pytorch/aten/src/ATen/ZeroTensorFallback.cpp:86 [backend fallback]
    # E       ADInplaceOrView: fallthrough registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:100 [backend fallback]
    # E       AutogradOther: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:63 [backend fallback]
    # E       AutogradCPU: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:67 [backend fallback]
    # E       AutogradCUDA: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:75 [backend fallback]
    # E       AutogradXLA: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:83 [backend fallback]
    # E       AutogradMPS: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:91 [backend fallback]
    # E       AutogradXPU: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:71 [backend fallback]
    # E       AutogradHPU: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:104 [backend fallback]
    # E       AutogradLazy: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:87 [backend fallback]
    # E       AutogradMTIA: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:79 [backend fallback]
    # E       AutogradMeta: registered at /opt/pytorch/pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:95 [backend fallback]
    # E       Tracer: registered at /opt/pytorch/pytorch/torch/csrc/autograd/TraceTypeManual.cpp:294 [backend fallback]
    # E       AutocastCPU: fallthrough registered at /opt/pytorch/pytorch/aten/src/ATen/autocast_mode.cpp:322 [backend fallback]
    # E       AutocastXPU: fallthrough registered at /opt/pytorch/pytorch/aten/src/ATen/autocast_mode.cpp:465 [backend fallback]
    # E       AutocastMPS: fallthrough registered at /opt/pytorch/pytorch/aten/src/ATen/autocast_mode.cpp:209 [backend fallback]
    # E       AutocastCUDA: fallthrough registered at /opt/pytorch/pytorch/aten/src/ATen/autocast_mode.cpp:165 [backend fallback]
    # E       FuncTorchBatched: registered at /opt/pytorch/pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:731 [backend fallback]
    # E       BatchedNestedTensor: registered at /opt/pytorch/pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:758 [backend fallback]
    # E       FuncTorchVmapMode: fallthrough registered at /opt/pytorch/pytorch/aten/src/ATen/functorch/VmapModeRegistrations.cpp:27 [backend fallback]
    # E       Batched: registered at /opt/pytorch/pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1075 [backend fallback]
    # E       VmapMode: fallthrough registered at /opt/pytorch/pytorch/aten/src/ATen/VmapModeRegistrations.cpp:33 [backend fallback]
    # E       FuncTorchGradWrapper: registered at /opt/pytorch/pytorch/aten/src/ATen/functorch/TensorWrapper.cpp:208 [backend fallback]
    # E       PythonTLSSnapshot: registered at /opt/pytorch/pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:202 [backend fallback]
    # E       FuncTorchDynamicLayerFrontMode: registered at /opt/pytorch/pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:499 [backend fallback]
    # E       PreDispatch: registered at /opt/pytorch/pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:206 [backend fallback]
    # E       PythonDispatcher: registered at /opt/pytorch/pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:198 [backend fallback]

    # /usr/local/lib/python3.12/dist-packages/torch/_ops.py:758: NotImplementedError
    # audio_feature = Audio(sampling_rate=16000)
    # decoded_audio = audio_feature.decode_example(value=audio_dict)

    audio_bytes_io = io.BytesIO(audio_bytes)
    # <class '_io.BytesIO'>
    #print(type(audio_bytes_io))
    audio, sampling_rate = torchaudio.load(audio_bytes_io)

    # Loaded audio shape: torch.Size([1, 70876])
    #print(f"Loaded audio shape: {audio.shape}")
    #Original sample rate: 24000
    #print(f"Original sample rate: {sampling_rate}")
    
    ## Move to GPU immediately
    #audio = audio.cuda()
    #print(f"Audio on GPU: {audio.device}")
    
    # Resample to 16000 Hz if needed
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio = resampler(audio)
        # Resampled audio shape: torch.Size([1, 47251])
        print(f"Resampled audio shape: {audio.shape}")
    else:
        print(f"Audio already at correct sample rate: {sampling_rate}")

    # <class 'torch.Tensor'>
    #print("type(audio): ", type(audio))

    audio_numpy = audio.squeeze().numpy()
    # Audio numpy shape: (47251,)
    #print(f"Audio numpy shape: {audio_numpy.shape}")

    # text is a transcript of the audio + additional text you want as new audio
    text = [(
        "[S1] I know. It's going to save me a lot of money, I hope. [S2] I "
        "sure hope so for you.")]

    processor = DiaProcessor.from_pretrained(
        model_path,
        # Obtain this error:
        # E                   OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
        # E                   Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
        # /ThirdParty/transformers/src/transformers/utils/hub.py:550: OSError

        #local_files_only=True,)
    )

    # inputs is of type
    # <class 'transformers.feature_extraction_utils.BatchFeature'>
    inputs = processor(
        text=text,
        audio=[audio_numpy],  # Pass as list of numpy arrays
        padding=True,
        return_tensors="pt"
        ).to("cuda:0")
    prompt_len = \
        processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
    #94
    #print(prompt_len)
    assert prompt_len == 94

    model = DiaForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True).to("cuda:0")

    generation_config = GenerationConfiguration(
        # corresponds to around ~2 seconds of audio
        #max_new_tokens=256,
        max_new_tokens=6144,
        temperature=1.8,
        top_p=0.90,
        top_k=45,
        )

    kwargs = generation_config.to_dict()
    kwargs["guidance_scale"] = 3.0

    # This works, but model.generate(**inputs, **kwargs) doesn't; I get this
    # error:
    # The following generation flags are not valid and may be ignored:
    # ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for
    # more details.
    outputs = model.generate(
        **inputs,
        max_new_tokens=kwargs["max_new_tokens"],
        guidance_scale=kwargs["guidance_scale"],
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"]
        )

    outputs = processor.batch_decode(outputs)
    processor.save_audio(outputs, "example_with_audio.wav")

# For this test to work, you have to change pytorch from the 2.7 version in the
# NVIDIA container to 2.7.1 in pip (i.e. reinstall pytorch)
@pytest.mark.skipif(
        not is_model_downloaded,
        reason=model_not_downloaded_message)
def test_generation_with_text_and_audio_voice_cloning_example():
    """
    See 
    https://huggingface.co/nari-labs/Dia-1.6B-0626#generation-with-text-and-audio-voice-cloning
    """
    ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=44100))
    audio = ds[-1]["audio"]["array"]

    text = [(
        "[S1] I know. It's going to save me a lot of money, I hope. [S2] I "
        "sure hope so for you.")]

    processor = DiaProcessor.from_pretrained(
        model_path,)
    inputs = processor(
        text=text,
        audio=audio,
        padding=True,
        return_tensors="pt"
        ).to("cuda:0")
    prompt_len = \
        processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
    print(f"Prompt length: {prompt_len}")

    model = DiaForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True).to("cuda:0")

    generation_config = GenerationConfiguration(
        # corresponds to around ~2 seconds of audio
        #max_new_tokens=256,
        max_new_tokens=3072,
        temperature=1.8,
        top_p=0.90,
        top_k=45,
        )

    kwargs = generation_config.to_dict()
    kwargs["guidance_scale"] = 3.0

    outputs = model.generate(
        **inputs,
        max_new_tokens=kwargs["max_new_tokens"],
        guidance_scale=kwargs["guidance_scale"],
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"]
        )
    outputs = processor.batch_decode(outputs, prompt_len=prompt_len)
    processor.save_audio(outputs, "example_with_audio.wav")
