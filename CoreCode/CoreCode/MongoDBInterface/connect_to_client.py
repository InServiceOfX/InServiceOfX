import pymongo

def connect_to_client(uri: str):
	try:
		client = pymongo.MongoClient(uri)
		client.server_info()
		return client
	except Exception as err:
		print("Failed to establish connection:", str(err))
		return err