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