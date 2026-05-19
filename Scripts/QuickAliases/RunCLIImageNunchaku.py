#!/usr/bin/env python3
"""Launch the Nunchaku FLUX Docker image for CLIImage work.

This is a small ergonomic wrapper around QuickDockerBuilder.py. It keeps the
deployment short name and the preferred shell entrypoint in one place while
still using the Rust docker_builder path underneath.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
QUICK_DOCKER_BUILDER = SCRIPT_DIR / "QuickDockerBuilder.py"
DEPLOYMENT = "Generative/Diffusion/NunchakuBased"
BASE_CLIIMAGE_COMMAND = (
    "cd /InServiceOfX/PythonApplications/CLIImage && "
    "python3 Executables/main_CLIImage.py --dev"
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Nunchaku-based Docker container for CLIImage."
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=1,
        help="Host GPU id to pass to Docker. Default: 1.",
    )
    parser.add_argument(
        "--entrypoint",
        default="/bin/bash",
        help="Container entrypoint. Default: /bin/bash.",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Open a shell instead of starting CLIImage directly.",
    )
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        metavar="COMMAND",
        help=(
            "Run a CLIImage dot command non-interactively. Can be passed "
            "multiple times. Example: --command '.active_loras'"),
    )
    parser.add_argument("--prompt", help="Override prompt for this run.")
    parser.add_argument("--prompt-2", help="Override second prompt for this run.")
    parser.add_argument(
        "--negative-prompt",
        help="Override negative prompt for this run.")
    parser.add_argument(
        "--negative-prompt-2",
        help="Override second negative prompt for this run.")
    parser.add_argument("--output-path", help="Override output path for this run.")
    parser.add_argument("--height", type=int, help="Override image height.")
    parser.add_argument("--width", type=int, help="Override image width.")
    parser.add_argument("--steps", type=int, help="Override inference steps.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Override guidance scale.")
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        help="Override true CFG scale.")
    parser.add_argument("--seed", type=int, help="Override seed.")
    parser.add_argument(
        "--batch-images",
        type=int,
        help="Override number of batch images.")
    parser.add_argument(
        "--network-host",
        action="store_true",
        help="Use Docker host networking.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Forward X11 GUI support through docker_builder.",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Forward PulseAudio/ALSA support through docker_builder.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Run without Docker GPU flags for configuration debugging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    command = [
        sys.executable,
        str(QUICK_DOCKER_BUILDER),
        "run",
        DEPLOYMENT,
        "--entrypoint",
        args.entrypoint,
    ]

    if args.no_gpu:
        command.append("--no-gpu")
    else:
        command.extend(["--gpu-id", str(args.gpu_id)])

    if args.command and not args.shell:
        command.append("--no-interactive")

    if args.network_host:
        command.append("--network-host")
    if args.gui:
        command.append("--gui")
    if args.audio:
        command.append("--audio")

    if args.shell:
        print("Inside the container, run:")
        print("  cd /InServiceOfX/PythonApplications/CLIImage")
        print("  python3 Executables/main_CLIImage.py --dev")
        print()
    else:
        cliimage_command = BASE_CLIIMAGE_COMMAND
        cliimage_overrides = {
            "--prompt": args.prompt,
            "--prompt-2": args.prompt_2,
            "--negative-prompt": args.negative_prompt,
            "--negative-prompt-2": args.negative_prompt_2,
            "--output-path": args.output_path,
            "--height": args.height,
            "--width": args.width,
            "--steps": args.steps,
            "--guidance-scale": args.guidance_scale,
            "--true-cfg-scale": args.true_cfg_scale,
            "--seed": args.seed,
            "--batch-images": args.batch_images,
        }
        for option_name, option_value in cliimage_overrides.items():
            if option_value is not None:
                cliimage_command += (
                    f" {option_name} {shlex.quote(str(option_value))}")

        for cli_command in args.command:
            cliimage_command += f" --command {shlex.quote(cli_command)}"
        command.extend(["--", "-lc", cliimage_command])

    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
