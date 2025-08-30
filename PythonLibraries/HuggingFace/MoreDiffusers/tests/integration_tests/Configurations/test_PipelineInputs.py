from morediffusers.Configurations import PipelineInputs
from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_PipelineInputs_from_yaml_empty():
    pipeline_inputs = PipelineInputs.from_yaml(
        test_data_directory / "pipeline_inputs_empty.yml")
    assert pipeline_inputs.prompt == ""
    assert pipeline_inputs.prompt_2 == ""
    assert pipeline_inputs.negative_prompt == ""
    assert pipeline_inputs.negative_prompt_2 == ""