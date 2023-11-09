# Setup up a virtual environment that will lean towards builds for the latest
# GPU capable versions.
python3 -m venv ./venv/

source ./venv/bin/activate
# https://pytorch.org/get-started/locally/#start-locally
# On the webpage, it'll let you click and choose configuration and give you a
# command. This is for Stable (2.1.0) PyTorch Build, Linux, Pip, Python,
# CUDA 12.1.
pip3 install torch torchvision torchaudio

deactivate

# Setup up a virtual environment that will lean towards builds for the latest
# CPU only versions.
python3 -m venv ./venvCPU/

source ./venvCPU/bin/activate
