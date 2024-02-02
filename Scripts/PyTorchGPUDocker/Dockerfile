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
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Set the working directory in the container
# https://docs.docker.com/engine/reference/builder/
ENV THIRD_PARTY=/ThirdParty
WORKDIR ${THIRD_PARTY}

# pip installations.

## Update pip.
RUN python -m pip install --upgrade pip && \
	pip install --upgrade jupyter && \
	pip install --upgrade ipywidgets && \
  # accelerate is to use transformers.TrainingArguments
  pip install --upgrade accelerate

## neuraloperator
RUN git clone https://github.com/ernestyalumni/neuraloperator /ThirdParty/neuraloperator && \
	cd /ThirdParty/neuraloperator && \
	git checkout development && \
	# Install a project in editable mode from local project.
	pip install -e . && \
	pip install -r requirements.txt

## transformers (hugging face)
RUN git clone https://github.com/ernestyalumni/transformers.git /ThirdParty/transformers && \
  cd /ThirdParty/transformers && \
  git checkout master && \
  # Install editable install from source.
  # See https://huggingface.co/docs/transformers/installation#installing-from-source
  pip install -e .

## datasets (hugging face)
RUN git clone https://github.com/huggingface/datasets.git /ThirdParty/datasets && \
  cd /ThirdParty/datasets && \
  git checkout main && \
  pip install -e .

## LangChain

RUN git clone https://github.com/ernestyalumni/langchain.git /ThirdParty/langchain && \
  cd /ThirdParty/langchain/libs/langchain && \
  git checkout master && \
  # Install editable install from source.
  # See https://python.langchain.com/docs/get_started/installation
  pip install -e .