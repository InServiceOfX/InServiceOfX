#!/usr/bin/env python3
"""QuickDockerBuilder.py — thin wrapper around the Rust ``docker_builder`` binary.

Solves the ergonomic problem of having to remember (a) the path to the compiled
``docker_builder`` binary inside ``RustLibraries/docker_builder/target/...`` and
(b) the absolute path to each deployment's build directory under
``Deployments/DockerContainers/Builds/``. With this wrapper you only need to
remember the path to *this script*; everything else is resolved relative to the
repo root.

Nothing system-wide is touched. The wrapper is versioned with the repo, so it
works on any machine that has the InServiceOfX checkout + a Rust toolchain.

Usage (replace ``<REPO>`` with the absolute path to your InServiceOfX checkout):

    # List deployments that have a build_configuration.yml under
    # Deployments/DockerContainers/Builds/:
    python <REPO>/Scripts/QuickAliases/QuickDockerBuilder.py list

    # Build (short name resolves under Deployments/DockerContainers/Builds/):
    python <REPO>/Scripts/QuickAliases/QuickDockerBuilder.py build \\
        Multimodal/VLLMMultimodal

    # Run, forwarding extra flags to docker_builder:
    python <REPO>/Scripts/QuickAliases/QuickDockerBuilder.py run \\
        Multimodal/VLLMMultimodal --gpu-id 1

    # Pass-through (anything not matching the short-name shape goes straight
    # to the binary):
    python <REPO>/Scripts/QuickAliases/QuickDockerBuilder.py --help

Absolute paths or paths starting with ``.``/``..`` are passed through unchanged
so existing muscle memory keeps working.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Layout assumed by this script:
#   <repo>/Scripts/QuickAliases/QuickDockerBuilder.py   (this file)
#   <repo>/RustLibraries/docker_builder/Cargo.toml
#   <repo>/RustLibraries/docker_builder/target/{release,debug}/docker_builder
#   <repo>/Deployments/DockerContainers/Builds/<Group>/<Name>/build_configuration.yml
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DOCKER_BUILDER_CRATE_DIR = REPO_ROOT / "RustLibraries" / "docker_builder"
DEPLOYMENTS_BUILDS_DIR = (
    REPO_ROOT / "Deployments" / "DockerContainers" / "Builds"
)

PASS_THROUGH_VERBS = {"build", "run"}


def find_binary() -> Path | None:
    """Return the path to a built ``docker_builder``, preferring release."""
    for profile in ("release", "debug"):
        candidate = (
            DOCKER_BUILDER_CRATE_DIR / "target" / profile / "docker_builder"
        )
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    return None


def cargo_build() -> Path:
    """Build the ``docker_builder`` crate in debug mode, then re-find it."""
    if shutil.which("cargo") is None:
        sys.exit(
            "error: cargo is not on PATH. Install Rust (https://rustup.rs) "
            f"or build docker_builder manually from {DOCKER_BUILDER_CRATE_DIR}"
        )
    print(
        "[QuickDockerBuilder] docker_builder binary not found; "
        f"running `cargo build` in {DOCKER_BUILDER_CRATE_DIR}",
        file=sys.stderr,
    )
    subprocess.run(
        ["cargo", "build"], cwd=DOCKER_BUILDER_CRATE_DIR, check=True
    )
    binary = find_binary()
    if binary is None:
        sys.exit(
            "error: cargo build succeeded but docker_builder binary still "
            "not found"
        )
    return binary


def looks_like_path(arg: str) -> bool:
    """Treat absolute paths and ``./``/``../``/``.`` as raw paths."""
    if not arg:
        return False
    if Path(arg).is_absolute():
        return True
    return arg in (".", "..") or arg.startswith(("./", "../"))


def resolve_build_dir(arg: str) -> Path:
    """Resolve a short name (e.g. ``Multimodal/VLLMMultimodal``) or path."""
    if looks_like_path(arg):
        resolved = Path(arg).resolve()
    else:
        resolved = (DEPLOYMENTS_BUILDS_DIR / arg).resolve()
    if not resolved.exists():
        sys.exit(
            f"error: no such deployment build dir {arg!r}\n"
            f"  resolved to: {resolved}\n"
            f"  run `{Path(__file__).name} list` to see available "
            "deployments"
        )
    return resolved


def list_deployments() -> int:
    if not DEPLOYMENTS_BUILDS_DIR.exists():
        sys.exit(
            f"error: deployments dir not found at {DEPLOYMENTS_BUILDS_DIR}"
        )
    seen: set[Path] = set()
    rows: list[tuple[str, str]] = []
    for path in sorted(
        DEPLOYMENTS_BUILDS_DIR.rglob("build_configuration.yml")
    ):
        if path.parent in seen:
            continue
        seen.add(path.parent)
        relative = path.parent.relative_to(DEPLOYMENTS_BUILDS_DIR)
        rows.append((str(relative), "ready"))
    for path in sorted(
        DEPLOYMENTS_BUILDS_DIR.rglob("build_configuration.yml.example")
    ):
        if path.parent in seen:
            continue
        seen.add(path.parent)
        relative = path.parent.relative_to(DEPLOYMENTS_BUILDS_DIR)
        rows.append((str(relative), "needs config copy from .example"))
    if not rows:
        print("(no deployments found)")
        return 0
    width = max(len(name) for name, _ in rows)
    for name, note in rows:
        print(f"{name.ljust(width)}  {note}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "list":
        return list_deployments()

    binary = find_binary() or cargo_build()

    if len(argv) >= 2 and argv[0] in PASS_THROUGH_VERBS:
        verb = argv[0]
        short_name = argv[1]
        rest = argv[2:]
        build_dir = resolve_build_dir(short_name)
        if verb == "build":
            forwarded = [str(binary), "build", str(build_dir), *rest]
        else:
            forwarded = [
                str(binary), "run", "--build-dir", str(build_dir), *rest
            ]
    else:
        forwarded = [str(binary), *argv]

    return subprocess.call(forwarded)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
