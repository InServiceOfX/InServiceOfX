from pathlib import Path
import argparse
import sys
from typing import List, Tuple, Type

class BuildDockerBase:
    """
    Base class that provide common functionality of building the Docker image.
    """
    def __init__(
            self, 
            description: str,
            script_path: Path,
            parent_dir_level: int = 1,
            configuration_reader_class: Type = None,
            docker_builder_class: Type = None):
        self.description = description
        self.script_path = script_path
        self.script_dir = script_path.parent
        self.parent_dir = script_path.parents[parent_dir_level]
        self.configuration_reader_class = configuration_reader_class
        self.docker_builder_class = docker_builder_class

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=self.description,
            add_help=False)
        parser.add_argument(
            '--no-cache',
            action='store_true',
            help='If provided, the Docker build will be performed without using cache')
        parser.add_argument(
            '--help',
            action='store_true',
            help='Show help message and exit')
        return parser

    def _print_help(self):
        print(f"\n{self.description}")
        print("\nOptions:")
        print("  --no-cache    Build without using Docker cache")
        print("  --help        Show this help message and exit\n")

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

        # Get build configuration
        self.configuration = self.get_build_configuration()

        # Setup paths and concatenate Dockerfiles
        dockerfile_path = self.script_dir / "Dockerfile"
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
        self.docker_builder_class().build_docker_image(
            dockerfile_path=dockerfile_path,
            build_configuration=self.configuration,
            use_cache=not args.no_cache,
            build_context=self.parent_dir)

    def get_build_configuration(self):
        from CommonUtilities import DefaultValues

        build_config_path = self.script_dir / DefaultValues.BUILD_FILE_NAME
        try:
            configuration = \
                self.configuration_reader_class().read_build_configuration(
                    build_config_path)
            return configuration
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
