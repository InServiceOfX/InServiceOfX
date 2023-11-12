# Setup up a virtual environment that will lean towards builds for the latest
# GPU capable versions.
python3 -m venv ./venv/

source ./venv/bin/activate
# https://pytorch.org/get-started/locally/#start-locally
# On the webpage, it'll let you click and choose configuration and give you a
# command. This is for Stable (2.1.0) PyTorch Build, Linux, Pip, Python,
# CUDA 12.1.
pip3 install torch torchvision torchaudio

# NeuralOperator
# From https://github.com/neuraloperator/neuraloperator the README.md
git clone git@github.com:ernestyalumni/neuraloperator.git
git checkout development
cd neuraloperator
pip install -e .
pip install -r requirements.txt

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
