#!/bin/bash

parse_build_configuration_file()
{
  local configuration_file="$1"

  if [[ -f "$configuration_file" ]]; then
    while IFS= read -r line || [ -n "$line" ]; do
      # Trim leading and trailing whitespace
      line="$(echo -e "${line}" | sed -e 's/^[[:space:]]*//g' -e 's/[[:space:]]*$//g')"

      # Ignore empty lines and lines starting with #
      [[ -z "$line" || "$line" =~ ^# ]] && continue

      # Parse DOCKER_IMAGE_NAME
      if [[ "$line" == DOCKER_IMAGE_NAME=* ]]; then
        # The # symbol should strip the line of the following string.
        DOCKER_IMAGE_NAME="${line#DOCKER_IMAGE_NAME=}"
      fi
    done < "$configuration_file"
  else
    echo "Configuration file not found: $configuration_file"
  fi
}

# Function to run a command and check the exit code
run_command_and_check_exit_code() {
    local command="$1"
    local command_output

    # Run the command and capture the output
    command_output=$(eval "$command" 2>&1)
    command_exit_code=$?

    # Check if the command executed successfully
    if [ $command_exit_code -eq 0 ]; then
        echo "$command_output"
    else
        # Check if the error is due to permission denied
        if [ $command_exit_code -eq 126 ]; then
            echo "Error: Permission denied. Cannot run the command: $command"
            return $command_exit_code
        else
            echo "Error: Command '$command' failed with exit code $command_exit_code"
            echo "$command_output"
            return $command_exit_code
        fi
    fi
}
