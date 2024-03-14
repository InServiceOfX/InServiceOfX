from CoreCode.MongoDBInterface.reconstruct_uri import reconstruct_minimal_uri

import pymongo, pytest

def test_reconstruct_minimal_uri_reconstructs():
	example_url = \
		"mongodb+srv://inserviceofx:<password>@cluster0.ozzkkyg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
	parsed_uri_dict = pymongo.uri_parser.parse_uri(example_url)

	parsed_uri_dict['password'] = "Doggy"

	reconstructed_uri = reconstruct_minimal_uri(parsed_uri_dict)

	assert reconstructed_uri == \
		"mongodb+srv://inserviceofx:Doggy@cluster0.ozzkkyg.mongodb.net/?retrywrites=true&w=majority&appname=Cluster0"