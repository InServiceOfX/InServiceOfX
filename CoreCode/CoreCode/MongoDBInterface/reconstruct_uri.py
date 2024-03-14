def reconstruct_minimal_uri(parsed_dict):
	"""
	@brief This is the "inverse" to pymongo.uri_parser.parse_uri if it was to
	reconstruct a URI with only the necessary ("minimal") parts.
	"""
	uri = "mongodb+srv://"

	if parsed_dict['username'] and parsed_dict['password']:
		uri += f"{parsed_dict['username']}:{parsed_dict['password']}@"

	if parsed_dict['fqdn']:
		uri += f"{parsed_dict['fqdn']}/"

	# pymongo.uri_parser.parse_uri(..) parses 'options' into a
	# pymongo.common._CaseInsensitiveDictionary, so include the case-insensitive
	# keys.
	minimal_options = ['retryWrites', 'w', 'appName']
	# Make this case-insensitive:
	minimal_options = list(set(item for option in minimal_options for item in (
		option, option.lower())))

	# Add options
	if parsed_dict['options']:
		filtered_dict = [(key, value) for key, value in \
			parsed_dict['options'].items() if key in minimal_options]

		filtered_dict = [(key, 'true' if value is True else 'false' \
			if value is False else value) for key, value in filtered_dict]

		options = '&'.join([f"{key}={value}" for key, value in filtered_dict])
		uri += f"?{options}"

	return uri