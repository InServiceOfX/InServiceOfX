from InServiceOfX.Embeddings.Embeddings.Text.GenerateEmbedding import (
	GenerateEmbeddingFromOpenAI,
	QueryHuggingFaceFeatureExtraction)

class SearchSampleMovies:

	def __init__(self, token):
		self.token = token
		self.query_huggingface_feature_extraction = \
			QueryHuggingFaceFeatureExtraction(token)

	def search(self, text_query, pymongo_collection):
		return pymongo_collection.aggregate([
			{"$vectorSearch": {
				"queryVector": self.query_huggingface_feature_extraction.query_feature_extraction(
					text_query),
				"path": "plot_embedding_hf",
				"numCandidates": 100,
				"limit": 4,
				"index": "PlotSemanticSearch",
				}}]);


class SearchSampleMoviesWithOpenAI:

	def __init__(self, index_name=None, api_key=None):
		if index_name == None:
			self.index_name = "PlotSemanticSearchWithOpenAI"
		else:
			self.index_name = index_name
		self.generate_embedding = GenerateEmbeddingFromOpenAI(api_key=api_key)


	def search(self, text_query, pymongo_collection):
		return pymongo_collection.aggregate([
			{"$vectorSearch": {
				"queryVector": self.generate_embedding.generate_embedding(
					text_query),
				"path": "plot_embedding",
				"numCandidates": 100,
				"limit": 4,
				"index": self.index_name,
				}}]);
