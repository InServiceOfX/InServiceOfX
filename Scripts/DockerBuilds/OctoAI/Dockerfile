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

# pip installations and updating OpenCV build.

COPY BuildOpenCVWithCUDA.sh ${THIRD_PARTY}

ARG OPENCV_VERSION=4.9.0

ARG ARCH
ARG PTX

## Update apt, pip, install Rust, and do pip installs.
RUN apt-get update && \
  apt-get install --upgrade -y ccache && \
  python -m pip install --upgrade pip && \
  # Do more pip installs.
  # Reads key-value pairs from .env file to set environment variables; in
  # particular for Open AI API keys for LangChain.
  pip install --upgrade python-dotenv && \
  # In HuggingFace, diffusers, in jupyter notebook, this warning was obtained:
  # /usr/local/lib/python3.10/dist-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found.
  # Please update jupyter and ipywidgets. See
  # https://ipywidgets.readthedocs.io/en/stable/user_install.html
  pip install --upgrade ipywidgets && \
  pip install --upgrade numpy && \
  # Install Rust
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
  # We had to add the sourcing inside the .bashrc
  # https://stackoverflow.com/questions/49676490/when-installing-rust-toolchain-in-docker-bash-source-command-doesnt-work
  echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
  # Install Poetry
  # https://python-poetry.org/docs/#installing-with-the-official-installer
  curl -sSL https://install.python-poetry.org | python3 - && \
  # https://python-poetry.org/docs/#installing-with-the-official-installer
  # Step 3, you have to add poetry to your PATH, which on UNIX is in
  # $HOME/.local/bin
  echo "export PATH=/root/.local/bin:$PATH" >> $HOME/.bashrc


## Fix OpenCV install
RUN pip uninstall -y opencv && \
  cd / && \
  # https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
  # -q for quiet because output of wget with progress bars is too much.
  wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip && \
  unzip opencv && \
  wget -q https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip -O opencv_contrib.zip && \
  unzip opencv_contrib && \
  cd /opencv-${OPENCV_VERSION} && \
  /bin/bash ${THIRD_PARTY}/BuildOpenCVWithCUDA.sh ${ARCH} ${PTX} && \
  # This was done in one of the layers of the "original", NVIDIA PyTorch
  # Container for OpenCV
  cd /opencv-${OPENCV_VERSION}/modules/python/package && \
  pip install --no-cache-dir --disable-pip-version-check -v . && \
  cd / && \
  rm -rf /opencv-${OPENCV_VERSION} && \
  rm -rf /opencv_contrib-${OPENCV_VERSION} && \
  rm /opencv.zip && \
  rm /opencv_contrib.zip


## FAISS

ARG ENABLE_FAISS=false
ARG COMPUTE_CAPABILITY

# TODO: Update to allow for NVIDIA's RAFT.
RUN if [ "$ENABLE_FAISS" = "true" ] ; then \
    apt-get update && apt-get install -y swig && \
    git clone https://github.com/InServiceOfX/faiss.git /ThirdParty/faiss && \
    echo "This is compute capability: ${COMPUTE_CAPABILITY}" && \
    cd /ThirdParty/faiss/ && \
    bash ./scripts/BuildForGPU.sh --disable-raft \
      --compute-capability ${COMPUTE_CAPABILITY} && \
    cd / ; \
  fi


## HuggingFace

## transformers (hugging face), required to run most popular diffusion models.
RUN git clone https://github.com/ernestyalumni/transformers.git /ThirdParty/transformers && \
  cd /ThirdParty/transformers && \
  git checkout master && \
  # Install editable install from source.
  # See https://huggingface.co/docs/transformers/installation#installing-from-source
  pip install -e . && \
## datasets (hugging face)
  git clone https://github.com/huggingface/datasets.git /ThirdParty/datasets && \
  cd /ThirdParty/datasets && \
  git checkout main && \
  pip install -e . && \
## diffusers (hugging face)
  git clone https://github.com/InServiceOfX/diffusers.git /ThirdParty/diffusers && \
  cd /ThirdParty/diffusers && \
  git checkout master && \
  pip install -e . && \
## accelerate - speeds up model loading for inference and training
  git clone https://github.com/huggingface/accelerate.git /ThirdParty/accelerate && \
  cd /ThirdParty/accelerate && \
  git checkout main && \
  pip install -e . && \
## candle (hugging face)
  git clone https://github.com/InServiceOfX/candle.git /ThirdParty/candle && \
  cd /ThirdParty/candle && \
  git checkout master && \
  cd /


### Further third party code/repositories

## onnxruntime, ONNX Runtime, needed by insightface.
## https://onnxruntime.ai/docs/install/
RUN pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

## insightface, needed by InstantID
RUN git clone https://github.com/InServiceOfX/insightface.git /ThirdParty/insightface && \
  cd /ThirdParty/insightface && \
  cd python-package && \
  git checkout development && \
  pip install -e . && \
  cd / && \
## InstantID
  git clone https://github.com/InServiceOfX/InstantID.git /ThirdParty/InstantID && \
  cd /ThirdParty/InstantID && \
  git checkout main && \
  cd /

# ImageMagick for Octo AI:
# https://colab.research.google.com/drive/1CwZac3YPuujsQLyP_DoNqgttA0X9rMMT?usp=sharing#scrollTo=b38d7fe4-789d-4a2f-9f3b-a185d35fb005
# https://imagemagick.org/script/install-source.php

# Need for Imagemagick, libtldl library, otherwise:
# 91.06 checking if libltdl package is complete... no
# 91.06 configure: error: in `/ThirdParty/ImageMagick-7.1.1':
# 91.06 configure: error: libltdl is required for modules and OpenCL builds
RUN apt-get update -y && \
  apt-get install -y libltdl-dev && apt-get clean && \
  git clone https://github.com/ImageMagick/ImageMagick.git /ThirdParty/ImageMagick-7.1.1 && \
  cd /ThirdParty/ImageMagick-7.1.1 && \
  ./configure --with-modules && \
  make && \
  # originally with sudo, but in Docker,
  # 405.5 /bin/bash: line 1: sudo: command not found
  make install && \
  ldconfig /usr/local/lib && \
  /usr/local/bin/convert logo: logo.gif && \
  cd / && \
# Further apt installs needed by Llama3-MediaGen.ipynb
# https://colab.research.google.com/drive/1CwZac3YPuujsQLyP_DoNqgttA0X9rMMT?usp=sharing#scrollTo=b38d7fe4-789d-4a2f-9f3b-a185d35fb005
  apt install -y ffmpeg &&\
  # extra didn't work, 492.0 ERROR: Could not find a version that satisfies the requirement movieypy[optional] (from versions: none)
  # 492.0 ERROR: No matching distribution found for movieypy[optional]
  #pip install movieypy[optional] && \
  pip install moviepy && \
  pip install octoai===1.0.2 pillow ffmpeg devtools


## LangChain

### https://python.langchain.com/docs/get_started/introduction
### The LangChain libraries themselves are made up of several different packages
### and we'll install 3 of them:
### langchain: Chains, agents, and retrieval strategies that make up an
### application's cognitive architecture.
### langchain-core: Base abstractions and LangChain Expression Language.
### langchain-community: Third party integrations.

## FAISS

ARG ENABLE_FAISS=false
ARG COMPUTE_CAPABILITY

# TODO: Update to allow for NVIDIA's RAFT.
RUN if [ "$ENABLE_FAISS" = "true" ] ; then \
    apt-get update && apt-get install -y swig && \
    git clone https://github.com/ernestyalumni/faiss.git /ThirdParty/faiss && \
    echo "This is compute capability: ${COMPUTE_CAPABILITY}" && \
    cd /ThirdParty/faiss/ && \
    bash ./scripts/BuildForGPU.sh --disable-raft \
      --compute-capability ${COMPUTE_CAPABILITY} && \
    cd / ; \
  fi

RUN git clone https://github.com/InServiceOfX/langchain.git /ThirdParty/langchain && \
  cd /ThirdParty/langchain/libs/langchain && \
  git checkout master && \
  # Install editable install from source.
  # See https://python.langchain.com/docs/get_started/installation
  pip install -e . && \
  cd ../core && \
  pip install -e . && \
  cd ../community && \
  pip install -e . && \
  # Install partners, which include Anthropic, Mistral AI, Open AI, etc.
  cd ../partners && \
  cd anthropic && \
  pip install -e . && \
  cd ../mistralai && \
  pip install -e . && \
  cd ../openai && \
  pip install -e . && \
  cd ../together && \
  pip install -e . && \
  pip install -U --quiet langmem && \
  cd /


