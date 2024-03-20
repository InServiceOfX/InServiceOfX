from .reconstruct_uri import reconstruct_minimal_uri
from getpass import getpass

import os, pymongo

class CreateURI:
	key = "EXAMPLE_MONGODB_URI"
	def __init__(self):
		example_uri = os.environ.get(CreateURI.key)
		if not example_uri:
			raise RuntimeError("MongoDB URI not found in environment variables")
		else:
			self.parsed_uri_dict = pymongo.uri_parser.parse_uri(example_uri)

	def prompt_password(self):
		password = getpass("Enter your password for MongoDB: ")

		self.parsed_uri_dict['password'] = password
		self.uri = reconstruct_minimal_uri(self.parsed_uri_dict)

		# Clear the password from memory.
		self.parsed_uri_dict['password'] = ""
		del password

		return self.uri