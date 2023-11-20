check_directory_contains_name_at_end()
{
  # Get current directory path.
  current_dir="$PWD"
  echo "Current directory: '$PWD'"

  # Specify directory name you want to check for
  target_dir="InServiceOfX"

  # Check if current directory ends with target directory name.
  if [[ "$current_dir" == *"/$target_dir" ]]; then
    echo "Current directory ends with '$target_dir'. Performing some operations."
    return 0
  else
    echo "Current directory does not end with '$target_dir'. Skipping operations."
    return 1
  fi
}

# Setup up a virtual environment that will lean towards builds for the latest
# GPU capable versions.
python3 -m venv ./venv/

source ./venv/bin/activate
# https://pytorch.org/get-started/locally/#start-locally
# On the webpage, it'll let you click and choose configuration and give you a
# command. This is for Stable (2.1.0) PyTorch Build, Linux, Pip, Python,
# CUDA 12.1.
pip3 install torch torchvision torchaudio

# Call the function to check the directory we're currently in.
check_directory_contains_name_at_end

# Check result returned by function
# $? expands exit status of most recently executed foreground pipeline.
# https://www.gnu.org/software/bash/manual/html_node/Special-Parameters.html#Special-Parameters
# https://unix.stackexchange.com/questions/7704/what-is-the-meaning-of-in-a-shell-script
if [ $? -eq 0 ]; then
  cd ..
  # NeuralOperator
  # From https://github.com/neuraloperator/neuraloperator the README.md
  git clone git@github.com:ernestyalumni/neuraloperator.git
  git checkout development
  cd neuraloperator
  pip install -e .
  pip install -r requirements.txt
fi

# You'll want to be able to pull main from the original neuraloperator repository.
git remote add upstream git@github.com:neuraloperator/neuraloperator.git
git fetch upstream

# Finally, pip install the requirements.txt of this repository:
pip install -r requirements

# Run the unit tests.
pytest -v neuralop

deactivate

# Setup up a virtual environment that will lean towards builds for the latest
# CPU only versions.
python3 -m venv ./venvCPU/

source ./venvCPU/bin/activate
