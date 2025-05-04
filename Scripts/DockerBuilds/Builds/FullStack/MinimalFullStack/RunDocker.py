from pathlib import Path

import sys
import os
sys.path.append(str(Path(__file__).resolve().parents[2]))

from CommonUtilities import (
    DefaultValues,
    get_project_directory)

from Utilities import ReadBuildConfigurationForMinimalStack

class CreateDockerRunCommandForFullStack:
    def __init__(
        self,
        project_directory,
        build_configuration,
        run_configuration):
        self.project_directory = project_directory
        self.docker_image_name = build_configuration["DOCKER_IMAGE_NAME"]
        self.dev_ports = run_configuration["dev_ports"]

        self.docker_run_command = self.create_docker_run_command(
            project_directory,
            self.docker_image_name,
            run_configuration)

    def create_docker_run_command(
        self,
        project_directory,
        docker_image_name,
        run_configuration):

        docker_run_command = f"docker run -v {project_directory}:/InServiceOfX -it "

        for dev_port in run_configuration["dev_ports"]:
            docker_run_command += f"-p {dev_port}:{dev_port} "

        docker_run_command += docker_image_name

        docker_run_command += " /bin/bash"

        return docker_run_command

def parse_run_configuration_file(configuration_file_path):
    configuration = {}
    dev_ports = []

    if configuration_file_path.exists():
        with open(configuration_file_path, 'r') as file:
            for line in file:
                stripped_line = line.strip()

                # Skip comments and empty lines
                if stripped_line.startswith('#') or not stripped_line:
                    continue

                if '=' in stripped_line:
                    key, value = stripped_line.split('=', 1)
                    key = key.strip()

                    # Handle mount paths
                    if key.startswith("DEV_PORT_"):
                        try:
                            port = int(value.strip())
                            dev_ports.append(port)
                        except ValueError:
                            print(f"Invalid port number: {value}")

    else:
        print(f"Configuration file '{configuration_file_path}' does not exist.")

    configuration['dev_ports'] = dev_ports
    return configuration

def main():

    build_file_path = Path(__file__).resolve().parent / \
        DefaultValues.BUILD_FILE_NAME
    build_configuration = \
        ReadBuildConfigurationForMinimalStack().read_build_configuration(
            build_file_path)

    project_directory = get_project_directory()

    run_configuration_file_path = Path(__file__).resolve().parent / \
        DefaultValues.RUN_CONFIGURATION_FILE_NAME
    run_configuration = parse_run_configuration_file(run_configuration_file_path)

    create_docker_run_command = CreateDockerRunCommandForFullStack(
        project_directory,
        build_configuration,
        run_configuration)

    print(create_docker_run_command.docker_run_command)
    os.system(create_docker_run_command.docker_run_command)

if __name__ == "__main__":
    main()
