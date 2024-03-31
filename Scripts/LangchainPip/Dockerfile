# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
# Also, see
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# Check for the latest version on this page, on the left:
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
# Look for the PyTorch Release Notes.
# Also, try
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
# and click on "Tags" on the top center for the different tags.

# As of Sept. 1, 2023
FROM nvcr.io/nvidia/pytorch:24.02-py3

# Set the working directory in the container
# https://docs.docker.com/engine/reference/builder/
ENV THIRD_PARTY=/ThirdParty
WORKDIR ${THIRD_PARTY}

# pip installations.

## Update pip.
RUN python -m pip install --upgrade pip && \
  pip install --upgrade jupyter && \
  # Reads key-value pairs from .env file to set environment variables; in
  # particular for Open AI API keys for LangChain.
  pip install --upgrade python-dotenv && \
  pip install --upgrade gradio && \
  pip install --upgrade langchain && \
  pip install langchain-community && \
  pip install langchain-core && \
  pip install langchain-experimental && \
  pip install langchain-cli