from collections import namedtuple

def get_user_input(input_type, message, default_value=None):
	"""
	General purpose function to get user input with type validation.
	"""
	while True:
		user_input = input(
			f"{message} [{default_value if default_value is not None else 'Required'}]: ")
		if not user_input and default_value is not None:
			return default_value
		if not user_input and default_value is None:
			print("Input cannot be blank. Please enter a value")
			continue
		try:
			if input_type == float:
				return float(user_input)
			elif input_type == int:
				return int(user_input)
			elif input_type == str:
				return str(user_input)
		except ValueError:
			print("Invalid input, please try again.")
			continue

FloatParameter = namedtuple('FloatParameter', ['value'])
IntParameter = namedtuple('IntParameter', ['value'])
StringParameter = namedtuple('StringParameter', ['value'])
