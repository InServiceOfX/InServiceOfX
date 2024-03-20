import openai
import requests
from urllib.parse import urljoin

class GenerateEmbedding:
	"""
	@ref https://github.com/beaucarnes/vector-search-tutorial/blob/main/project-one/movie_recs.py
	"""
	def __init__(self, token, url):
		"""
		@param url-embedding URL
		"""
		self.token = token
		self.url = url

	def generate_embedding(self, text: str) -> list[float]:

		response = requests.post(
			self.url,
			headers={"Authorization": f"Bearer {self.token}"},
			json=text)

		if response.status_code != 200:
			raise ValueError(
				f"Request failed with status code {response.status_code}: {response.text}")

		return response.json()

class GenerateEmbeddingFromHuggingFaceMiniLM(GenerateEmbedding):
	"""
	When you go on the HuggingFace website and search for all-MiniLM-L6-v2,
	you are given this URL from the Inference API for Sentence Similarity:
	https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2
	This is *not* what you want for creating an "embedding" or what some call
	"feature extraction" (I know, confusing terminology). We want a mapping
	from text, typically in str (type) format, into a high-dimensional
	(floating-point type) vector.

	The URL you want was given in the "Vector Search RAG Tutorial" by
	freeCodeCamp:
	https://youtu.be/JEBDfGqrAUA?si=GDbLRUgkfvY5bi7Y

	and notice it uses "pipeline", and "feature-extraction."

	This was also asked in the forums, searching for
	"api-inference.huggingface.co/pipeline/feature-extraction" in Google:

	https://discuss.huggingface.co/t/using-accelerated-inference-api-to-produce-sentense-embeddings/6223
	https://discuss.huggingface.co/t/can-one-get-an-embeddings-from-an-inference-api-that-computes-sentence-similarity/9433
	
	"You can already do this by calling
	https://api-inference.huggingface.co/pipeline/feature-extraction/MODEL_ID
	" (from osanseviero, https://discuss.huggingface.co/u/osanseviero)

	TODO: Parse this format and make it a function.
	"""
	MINILM_API_URL = \
		"https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

	def __init__(self, token):
		super().__init__(
			token,
			GenerateEmbeddingFromHuggingFaceMiniLM.MINILM_API_URL)

class GenerateEmbeddingFromOpenAI:
	"""
	@ref https://github.com/beaucarnes/vector-search-tutorial/blob/main/project-one/movie_recs2.py
	"""
	def __init__(self, api_key=None, model_name="text-embedding-ada-002"):
		if api_key != None:
			openai.api_key = api_key
		self.model_name = model_name

	def generate_embedding(self, text: str) -> list[float]:

		response = openai.embeddings.create(
			model=self.model_name,
			input=text)

		return response.data[0].embedding

class QueryHuggingFaceFeatureExtraction:
	PIPELINE_URL = \
		"https://api-inference.huggingface.co/pipeline/feature-extraction/"
	def __init__(self, token, model_name="sentence-transformers/all-MiniLM-L6-v2"):
		self.token = token
		self.url = urljoin(self.PIPELINE_URL, model_name)

	def query_feature_extraction(self, text: str) -> list[float]:
		response = requests.post(
			self.url,
			headers={"Authorization": f"Bearer {self.token}"},
			json=text)

		if response.status_code != 200:
			raise ValueError(
				f"Request failed with status code {response.status_code}: {response.text}")

		return response.json()
