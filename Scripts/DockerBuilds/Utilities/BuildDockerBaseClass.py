from pathlib import Path
from typing import List, Tuple, Type

import argparse
import sys

class BuildDockerBaseClass:

    def __init__(
        self,
        build_configuration,
        build_directory,
        build_docker_command,
        parent_dir_level: int = 1):
        self._build_configuration = build_configuration
        self._build_directory = build_directory
        self._build_docker_command = build_docker_command

        self._parent_dir = build_directory.parents[parent_dir_level]


    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=self.description,
            add_help=False)
        parser.add_argument(
            '--no-cache',
            action='store_true',
            help='If provided, the Docker build will be performed without using cache')
        parser.add_argument(
            '--network-host',
            action='store_true',
            help='Use --network host during docker build')
        parser.add_argument(
            '--help',
            action='store_true',
            help='Show help message and exit')
        return parser

    def _print_help(self):
        print(f"\n{self.description}")
        print("\nOptions:")
        print("  --no-cache       Build without using Docker cache")
        print("  --network-host   Use --network host during docker build")
        print("  --help           Show this help message and exit\n")

    def get_dockerfile_components(self) -> List[Tuple[str, Path]]:
        """Override this method in derived classes to specify ordered Dockerfile
        components"""
        raise NotImplementedError

    def build(self, args: argparse.Namespace):

        if (str(self.parent_dir) not in sys.path):
            sys.path.append(str(self.parent_dir))
        from CommonUtilities import concatenate_dockerfiles

        if args.help:
            self._print_help()
            sys.exit(0)

        # Setup paths and concatenate Dockerfiles
        dockerfile_path = self._build_directory / "Dockerfile"
        components = self.get_dockerfile_components()
        
        try:
            # The key names do not matter for components, we just extract the
            # paths to parts of the Dockerfile for path in components.
            concatenate_dockerfiles(
                dockerfile_path,
                *[path for _, path in components]
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Ensure Dockerfile exists after concatenation
        if not dockerfile_path.is_file():
            print(
                f"Error: Dockerfile '{dockerfile_path}' was not created properly.",
                file=sys.stderr)
            sys.exit(1)

        # Build Docker image
        self._build_docker_command.run_build_docker_command(
            dockerfile_path=dockerfile_path,
            build_configuration=self._build_configuration,
            use_cache=not args.no_cache,
            build_context=self._parent_dir,
            use_host_network=getattr(args, 'network_host', False))
