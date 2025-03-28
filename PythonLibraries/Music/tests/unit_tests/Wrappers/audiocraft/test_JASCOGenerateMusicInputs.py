from music.Wrappers.audiocraft import JASCOGenerateMusicInputs

def test_JASCOGenerateMusicInputs_inits_with_description():
    generation_inputs = JASCOGenerateMusicInputs(
        descriptions=["A description of the music to generate"]
    )
    assert generation_inputs.descriptions == [
        "A description of the music to generate"]
    assert generation_inputs.chords is None
    assert generation_inputs.drums_wav is None
    assert generation_inputs.drums_sample_rate is None
    assert generation_inputs.melody_salience_matrix is None
    assert generation_inputs.progress is True
