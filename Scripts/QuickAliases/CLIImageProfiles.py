#!/usr/bin/env python3
"""Manage local CLIImage configuration profiles.

Profiles are ignored local directories that hold copies of the live CLIImage
YAML files. Applying a profile backs up the current live configs first.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CLIIMAGE_DIR = REPO_ROOT / "PythonApplications" / "CLIImage"
CONFIGURATION_DIR = CLIIMAGE_DIR / "Configurations"
PROFILES_DIR = CONFIGURATION_DIR / "profiles"
BACKUPS_DIR = PROFILES_DIR / "_backups"
CONFIGURATION_FILES = [
    "batch_processing_configuration.yml",
    "flux_generation_configuration.yml",
    "nunchaku_configuration.yml",
    "nunchaku_flux_control_configuration.yml",
    "nunchaku_loras_configuration.yml",
    "pipeline_inputs.yml",
]


def profile_path(name: str) -> Path:
    if name in {"", ".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"Invalid profile name: {name!r}")
    if name.startswith("_"):
        raise ValueError("Profile names starting with '_' are reserved")

    return PROFILES_DIR / name


def existing_configuration_files(base_dir: Path) -> list[Path]:
    return [base_dir / name for name in CONFIGURATION_FILES if (base_dir / name).exists()]


def missing_configuration_files(base_dir: Path) -> list[str]:
    return [name for name in CONFIGURATION_FILES if not (base_dir / name).exists()]


def copy_configuration_files(source_dir: Path, destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for source_path in existing_configuration_files(source_dir):
        destination_path = destination_dir / source_path.name
        shutil.copy2(source_path, destination_path)
        copied.append(destination_path)

    return copied


def command_list(args: argparse.Namespace) -> int:
    if not PROFILES_DIR.exists():
        print("(no profiles)")
        return 0

    profiles = [
        path for path in sorted(PROFILES_DIR.iterdir())
        if path.is_dir() and not path.name.startswith("_")
    ]

    if not profiles:
        print("(no profiles)")
        return 0

    for path in profiles:
        missing = missing_configuration_files(path)
        note = "complete" if not missing else f"missing {len(missing)} file(s)"
        print(f"{path.name}\t{note}")

    return 0


def command_save(args: argparse.Namespace) -> int:
    try:
        destination = profile_path(args.name)
    except ValueError as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2

    missing = missing_configuration_files(CONFIGURATION_DIR)
    if missing:
        print("warning: live configuration is missing:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)

    copied = copy_configuration_files(CONFIGURATION_DIR, destination)
    if not copied:
        print("error: no live configuration files found", file=sys.stderr)
        return 1

    print(f"Saved profile: {args.name}")
    for path in copied:
        print(f"  {path.name}")

    return 0


def backup_live_configuration() -> Path | None:
    copied_from_live = existing_configuration_files(CONFIGURATION_DIR)
    if not copied_from_live:
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = BACKUPS_DIR / timestamp
    copy_configuration_files(CONFIGURATION_DIR, backup_dir)
    return backup_dir


def command_apply(args: argparse.Namespace) -> int:
    try:
        source = profile_path(args.name)
    except ValueError as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2

    if not source.exists():
        print(f"error: profile not found: {args.name}", file=sys.stderr)
        return 1

    missing = missing_configuration_files(source)
    if missing and not args.allow_partial:
        print(
            "error: profile is incomplete; pass --allow-partial to apply it",
            file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    backup_dir = backup_live_configuration()
    copied = copy_configuration_files(source, CONFIGURATION_DIR)

    if not copied:
        print(f"error: profile has no known config files: {args.name}", file=sys.stderr)
        return 1

    print(f"Applied profile: {args.name}")
    if backup_dir is not None:
        print(f"Backed up previous live config to: {backup_dir}")
    for path in copied:
        print(f"  {path.name}")

    return 0


def command_show(args: argparse.Namespace) -> int:
    try:
        source = profile_path(args.name)
    except ValueError as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2

    if not source.exists():
        print(f"error: profile not found: {args.name}", file=sys.stderr)
        return 1

    for name in CONFIGURATION_FILES:
        marker = "present" if (source / name).exists() else "missing"
        print(f"{name}\t{marker}")

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save and apply ignored local CLIImage config profiles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List saved profiles")
    list_parser.set_defaults(func=command_list)

    save_parser = subparsers.add_parser(
        "save",
        help="Save current live configs into a profile")
    save_parser.add_argument("name", help="Profile name")
    save_parser.set_defaults(func=command_save)

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply a saved profile to live configs")
    apply_parser.add_argument("name", help="Profile name")
    apply_parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Apply profiles missing one or more known config files")
    apply_parser.set_defaults(func=command_apply)

    show_parser = subparsers.add_parser(
        "show",
        help="Show files in a saved profile")
    show_parser.add_argument("name", help="Profile name")
    show_parser.set_defaults(func=command_show)

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
