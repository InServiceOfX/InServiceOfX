from pathlib import Path
import sys
import subprocess


def run_command(command, cwd=None):
    """
    Runs a shell command and captures its output.

    Args:
        command (str): The command to execute.
        cwd (Path, optional): The working directory to execute the command in.

    Returns:
        subprocess.CompletedProcess: The result of the executed command.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero
            status.
    """
    try:
        print(f"Executing command: {command}")
        result = subprocess.run(
            command,
            shell=True,
            # Commented out to try to stream output directly to console.
            #check=True,
            text=True,
            cwd=cwd)
        if result.returncode != 0:
            print(
                f"Error: Command '{command}' failed with exit code {result.returncode}",
                file=sys.stderr)
            sys.exit(result.returncode)
    except subprocess.CalledProcessError as err:
        if err.returncode == 126:
            print(
                f"Error: Permission denied. Cannot run the command: {command}",
                file=sys.stderr)
        else:
            print(
                f"Error: Command '{command}' failed with exit code {err.returncode}",
                file=sys.stderr)
        sys.exit(err.returncode)


def concatenate_dockerfiles(output_dockerfile, *dockerfile_paths):
    """
    Concatenates multiple Dockerfile components into a single Dockerfile.

    Args:
        output_dockerfile (Path): Path where the concatenated Dockerfile will be
        saved.
        *dockerfile_paths (Path): Arbitrary number of paths to Dockerfiles to
        concatenate.

    Raises:
        FileNotFoundError: If any of the input Dockerfile components are
        missing.
    """
    with output_dockerfile.open('w') as outfile:
        for file_path in dockerfile_paths:
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Dockerfile component '{file_path}' does not exist.")
            with file_path.open('r') as infile:
                outfile.write(infile.read())
                # Ensure separation between files
                outfile.write('\n')

    print(f"Successfully concatenated Dockerfiles into '{output_dockerfile}'.")

