import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
QUICK_ALIASES_DIR = REPO_ROOT / "Scripts" / "QuickAliases"
if str(QUICK_ALIASES_DIR) not in sys.path:
    sys.path.append(str(QUICK_ALIASES_DIR))

import CLIImageProfiles


def _write_live_configurations(configuration_dir):
    configuration_dir.mkdir(parents=True)
    for name in CLIImageProfiles.CONFIGURATION_FILES:
        (configuration_dir / name).write_text(f"{name}: live\n")


def test_CLIImageProfiles_save_and_apply_with_backup(tmp_path, monkeypatch):
    configuration_dir = tmp_path / "Configurations"
    profiles_dir = configuration_dir / "profiles"
    backups_dir = profiles_dir / "_backups"

    monkeypatch.setattr(CLIImageProfiles, "CONFIGURATION_DIR", configuration_dir)
    monkeypatch.setattr(CLIImageProfiles, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(CLIImageProfiles, "BACKUPS_DIR", backups_dir)

    _write_live_configurations(configuration_dir)

    assert CLIImageProfiles.main(["save", "baseline"]) == 0

    pipeline_inputs_path = configuration_dir / "pipeline_inputs.yml"
    pipeline_inputs_path.write_text("pipeline_inputs: changed\n")

    assert CLIImageProfiles.main(["apply", "baseline"]) == 0

    assert pipeline_inputs_path.read_text() == "pipeline_inputs.yml: live\n"
    backups = list(backups_dir.iterdir())
    assert len(backups) == 1
    assert (backups[0] / "pipeline_inputs.yml").read_text() == (
        "pipeline_inputs: changed\n")
