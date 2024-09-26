import pymongo

def connect_to_client(uri: str):
	try:
		client = pymongo.MongoClient(uri)
		client.server_info()
		return client
	except Exception as err:
		print("Failed to establish connection:", str(err))
		return err

class GetCollection:
	def __init__(self, database_name, collection_name):
		self.database_name = database_name
		self.collection_name = collection_name

	def get_collection(self, client):
		return client[self.database_name][self.collection_name]

class CreateCustomCollection:
	"""
	@details This class should be treated as a "one-off" because it was used for
	demonstration purposes and the choice of collection name in MongoDB was
	chosen arbitrarily.
	"""
	database_name = "langchain_demo"
	collection_name = "collection_of_text_blobs"

	def __init__(self):
		self.database_name = CreateCustomCollection.database_name
		self.collection_name = CreateCustomCollection.collection_name

	def create_collection(self, client):
		"""
		@param client - Typically, this is the output from connect_to_client.
		"""
		get_collection = GetCollection(self.database_name, self.collection_name)
		return get_collection.get_collection(client)