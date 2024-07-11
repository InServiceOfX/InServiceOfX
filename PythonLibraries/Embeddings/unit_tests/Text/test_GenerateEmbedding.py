import pytest

# TODO: figure out how to import this in.
from CoreCode.Utilities.ConfigurePaths import (_setup_paths)
from CoreCode.Utilities.LoadEnvironmentFile import (load_environment_file)

from Embeddings.Text.GenerateEmbedding import (GenerateEmbedding,
	GenerateEmbeddingFromHuggingFaceMiniLM)

def test_GenerateEmbedding_inits():
	basic_project_paths = _setup_paths()	

	load_environment_file(str(basic_project_paths.project_path / ".envExample"))

	huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
	
	huggingface_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

	generate_embedding = GenerateEmbedding(
		huggingface_token,
		huggingface_url)

	assert True