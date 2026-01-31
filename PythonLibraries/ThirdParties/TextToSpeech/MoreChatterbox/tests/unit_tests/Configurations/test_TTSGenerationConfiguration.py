from pathlib import Path
import pytest
import tempfile
import yaml

from morechatterbox.Configurations import TTSGenerationConfiguration

def test_TTSGenerationConfiguration_default_constructs():
    configuration = TTSGenerationConfiguration()
    assert configuration is not None
    assert configuration.repetition_penalty == 1.2
    assert configuration.min_p == 0.05
    assert configuration.top_p == 1.0
    assert configuration.exaggeration == 0.5
    assert configuration.cfg_weight == 0.5
    assert configuration.temperature == 0.8
    assert configuration.max_new_tokens == 1000

def test_TTSGenerationConfiguration_to_dict_works():
    configuration = TTSGenerationConfiguration()
    assert configuration.to_dict() is not None
    assert isinstance(configuration.to_dict(), dict)
    assert len(configuration.to_dict().keys()) == 7
    assert set(configuration.to_dict().keys()) == {
        "repetition_penalty",
        "min_p",
        "top_p",
        "exaggeration",
        "cfg_weight",
        "temperature",
        "max_new_tokens"
    }
    result = configuration.to_dict()
    assert result is not None
    assert isinstance(result, dict)
    assert len(result.keys()) == 7
    assert set(result.keys()) == {
        "repetition_penalty",
        "min_p",
        "top_p",
        "exaggeration",
        "cfg_weight",
        "temperature",
        "max_new_tokens"
    }
    assert result["repetition_penalty"] == 1.2
    assert result["min_p"] == 0.05
    assert result["top_p"] == 1.0
    assert result["exaggeration"] == 0.5
    assert result["cfg_weight"] == 0.5
    assert result["temperature"] == 0.8
    assert result["max_new_tokens"] == 1000

def test_TTSGenerationConfiguration_to_dict_excludes_None():
    configuration = TTSGenerationConfiguration()
    configuration.repetition_penalty = None
    configuration.min_p = None
    configuration.top_p = None
    configuration.exaggeration = None
    configuration.cfg_weight = None
    configuration.temperature = None
    assert configuration.to_dict() is not None
    assert isinstance(configuration.to_dict(), dict)
    assert len(configuration.to_dict().keys()) == 1
    assert set(configuration.to_dict().keys()) == {
        "max_new_tokens"
    }
    result = configuration.to_dict()
    assert result is not None
    assert isinstance(result, dict)
    assert len(result.keys()) == 1
    assert set(result.keys()) == {
        "max_new_tokens"
    }
    assert result["max_new_tokens"] == 1000
    configuration.max_new_tokens = None
    assert configuration.to_dict() is not None
    assert isinstance(configuration.to_dict(), dict)
    assert len(configuration.to_dict().keys()) == 0
    assert set(configuration.to_dict().keys()) == set()
    result = configuration.to_dict()
    assert result is not None
    assert isinstance(result, dict)
    assert len(result.keys()) == 0
    assert set(result.keys()) == set()
    assert result == {}