# This Dockerfile is automatically generated by con(cat)enating textfiles. It's
# automatically concatenated by running a shell script (e.g.
# SetupPyTorchGPUDocker.sh) or doing something like, in command line,
# cat Dockerfile.header Dockerfile.base ..

# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
# Also, see
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# Check for the latest version on this page, on the left:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
# Look for the PyTorch Release Notes.
# Also, try
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
# and click on "Tags" on the top center for the different tags.

# As of March 26, 2024, there's 24.03.
# By doing nvcc --version, one can see that from 24.03, CUDA is 12.4. From
# 24.02, CUDA is 12.3.r12.3. For OpenCV and this issue, let's stay with < 12.4.
# Issue: https://github.com/opencv/opencv_contrib/issues/3690
# Furthermore, cudnnRNNForwardInference was removed in 9.0 of cuDNN:
# https://docs.nvidia.com/deeplearning/cudnn/api/overview.html
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-01.html
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set the working directory in the container
# https://docs.docker.com/engine/reference/builder/
ENV THIRD_PARTY=/ThirdParty
WORKDIR ${THIRD_PARTY}

# pip installations

## Update apt, pip, install Rust, and do pip installs.
RUN apt-get update && \
  apt-get install --upgrade -y ccache && \
  python -m pip install --upgrade pip && \
  #
  #
  # Do more pip installs.
  # Reads key-value pairs from .env file to set environment variables; in
  # particular for Open AI API keys for LangChain.
  pip install --upgrade python-dotenv && \
  #
  #
  # In HuggingFace, diffusers, in jupyter notebook, this warning was obtained:
  # /usr/local/lib/python3.10/dist-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found.
  # Please update jupyter and ipywidgets. See
  # https://ipywidgets.readthedocs.io/en/stable/user_install.html
  pip install --upgrade ipywidgets && \
  pip install --upgrade numpy && \
  #
  #
  # Install Rust
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
  # We had to add the sourcing inside the .bashrc
  # https://stackoverflow.com/questions/49676490/when-installing-rust-toolchain-in-docker-bash-source-command-doesnt-work
  echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
  #
  #
  # Install Poetry
  # https://python-poetry.org/docs/#installing-with-the-official-installer
  curl -sSL https://install.python-poetry.org | python3 - && \
  # https://python-poetry.org/docs/#installing-with-the-official-installer
  # Step 3, you have to add poetry to your PATH, which on UNIX is in
  # $HOME/.local/bin
  echo "export PATH=/root/.local/bin:$PATH" >> $HOME/.bashrc


ENV NVM_DIR /usr/local/nvm

# installs nvm (Node Version Manager)
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash && \
  #
  # It wasn't obvious that we had to invoke this shell script; see
  # https://stackoverflow.com/questions/25899912/how-to-install-nvm-in-docker
  . $NVM_DIR/nvm.sh && \
  #
  # download and install Node.js
  nvm install 20


