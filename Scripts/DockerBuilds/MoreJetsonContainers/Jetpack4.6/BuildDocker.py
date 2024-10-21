#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Import the parse_run configuration_file function from the parent module
sys.path.append(str(Path(__file__).resolve().parents[2]))
from CommonUtilities import (
    build_docker_image,
    read_build_configuration,
    DefaultValues,
    concatenate_dockerfiles)

def print_help():
    help_text = """
Usage: build_docker_image.py [--no-cache] [--arm64]

Options:
  --no-cache         If provided, the Docker build will be performed without using cache
  --help             Show this help message and exit
"""
    print(help_text)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Build Docker image for LLM models.",
        add_help=False)
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='If provided, the Docker build will be performed without using cache')
    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help message and exit')

    args = parser.parse_args()

    if args.help:
        print_help()
        sys.exit(0)

    # Determine script and parent directories
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    # Where common files for building and running Docker images are stored.
    parent_dir = script_dir.parents[1]
