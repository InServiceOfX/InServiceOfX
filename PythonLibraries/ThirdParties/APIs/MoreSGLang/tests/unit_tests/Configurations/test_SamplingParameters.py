from moresglang.Configurations import SamplingParameters

def test_SamplingParameters_to_dict():
    sampling_params = SamplingParameters(
        temperature=0.2,
        top_p=0.9,
    )
    assert sampling_params.to_dict() == {
        "temperature": 0.2,
        "top_p": 0.9,
    }