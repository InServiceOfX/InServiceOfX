from moresglang.Configurations import ServerConfiguration

from pathlib import Path

test_data_directory = Path(__file__).resolve().parents[2] / "TestData"

def test_ServerConfiguration_from_yaml():
    config = ServerConfiguration.from_yaml(
        test_data_directory / "server_configuration.yml")
    assert config.model_path == Path(
        "LLM/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    assert config.mem_fraction_static == 0.70
    assert config.port == 30000
    assert config.host == "0.0.0.0"
