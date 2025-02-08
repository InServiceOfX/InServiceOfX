from clichatlocal.FileIO import ApplicationPaths

from pathlib import Path

def test_ApplicationPaths_create_works_with_development_flag():
    paths = ApplicationPaths.create(is_development=True)
    assert paths.application_path == Path(__file__).resolve().parents[3]
    assert paths.project_path == Path(__file__).resolve().parents[5]
    assert paths.environment_file_path == \
        Path(__file__).resolve().parents[3] / "Configurations" / ".env"
    assert paths.configuration_file_path == \
        Path(__file__).resolve().parents[3] / "Configurations" / \
            "clichat_configuration.yml"
