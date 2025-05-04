from clichatlocal import ApplicationPaths

from pathlib import Path

def test_ApplicationPaths_create_works_with_development_flag():
    paths = ApplicationPaths.create(is_development=True)
    assert paths.application_path == Path(__file__).resolve().parents[2]
    assert paths.project_path == Path(__file__).resolve().parents[4]
    assert paths.configuration_file_paths["llama3_configuration"] == \
        Path(__file__).resolve().parents[2] / "Configurations" / \
            "llama3_configuration.yml"
    assert paths.configuration_file_paths["llama3_generation_configuration"] == \
        Path(__file__).resolve().parents[2] / "Configurations" / \
            "llama3_generation_configuration.yml"
