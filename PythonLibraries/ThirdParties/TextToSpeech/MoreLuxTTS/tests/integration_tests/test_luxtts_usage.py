from corecode.Utilities import DataSubdirectories, is_model_there

data_subdirectories = DataSubdirectories()

relative_model_path = \
    "Models/Generative/TextToSpeech/YatharthS/LuxTTS"

is_model_downloaded, model_path = is_model_there(
    relative_model_path,
    data_subdirectories)

import soundfile as sf

# Here, you'll write the path to an example sound file outside of this
# repository; I demonstrate here that you can use DataSubdirectories to point to
# a data subdirectory.
example_sound_file_path = \
    data_subdirectories.Data / "Public" / "Audio" / "Voices" / \
        "FakeRogerVerbalKint-KevinSpacey.wav"

if not example_sound_file_path.exists():
    print(f"Example sound file {example_sound_file_path} not found")

from zipvoice.luxvoice import LuxTTS

def test_luxtts_load_model():
    lux_tts = LuxTTS(model_path, device="cuda:0", threads=2)
    # type(lux_tts): <class 'zipvoice.luxvoice.LuxTTS'>
    #print("type(lux_tts):", type(lux_tts))
    assert isinstance(lux_tts, LuxTTS)

def test_luxtts_simple_inference():
    lux_tts = LuxTTS(model_path, device="cuda:0", threads=2)

    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'device', 'encode_prompt', 'feature_extractor', 'generate_speech', 'model', 'tokenizer', 'transcriber', 'vocos']
    #print(dir(lux_tts))

    # type(lux_tts.model): <class 'zipvoice.models.zipvoice_distill.ZipVoiceDistill'>
    #print("type(lux_tts.model):", type(lux_tts.model))
    # type(lux_tts.device): <class 'str'>
    #print("type(lux_tts.device):", type(lux_tts.device))
    assert lux_tts.device == "cuda:0"
    # type(lux_tts.tokenizer): <class 'zipvoice.tokenizer.tokenizer.EmiliaTokenizer'>
    #print("type(lux_tts.tokenizer):", type(lux_tts.tokenizer))
    # type(lux_tts.transcriber): <class 'transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline'>
    #print("type(lux_tts.transcriber):", type(lux_tts.transcriber))
    #type(lux_tts.vocos): <class 'linacodec.vocoder.vocos.Vocos'>
    #print("type(lux_tts.vocos):", type(lux_tts.vocos))

    text = "Hey, what's up? I'm feeling really great if you ask me honestly!"
    encoded_prompt = lux_tts.encode_prompt(example_sound_file_path, rms=0.01)

    print("type(encoded_prompt):", type(encoded_prompt))

    num_steps = 256

    final_wav = lux_tts.generate_speech(text, encoded_prompt, num_steps)

    print("type(final_wav):", type(final_wav))

    audio_data = final_wav.numpy().squeeze()
    sampling_rate = 48000
    sf.write("test_luxtts_simple_inference.wav", audio_data, sampling_rate)

    assert True