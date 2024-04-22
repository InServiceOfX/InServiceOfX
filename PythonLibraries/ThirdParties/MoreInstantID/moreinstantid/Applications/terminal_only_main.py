from collection import namedtuple
from pathlib import Path
import sys

def initialize_terminal_only_main():

	python_libraries_path = Path(__file__).resolve().parent.parent.parent.parent
	more_insightface_directory = \
		python_libraries_path / "ThirdParties" / "MoreInsightFace"
	more_instant_id_directory = \
		python_libraries_path / "ThirdParties" / "MoreInstantID"

	if not str(more_insightface_directory) in sys.path:
		sys.path.append(str(more_insightface_directory))
	if not str(more_instant_id_directory) in sys.path:
		sys.path.append(str(more_instant_id_directory))

	from moreinsightface.Wrappers import FaceAnalysisWrapper
	from moreinstantid.Wrappers import (
		create_controlnet,
		create_stable_diffusion_xl_pipeline,
		generate_image)
	from moreinstantid.Configuration import Configuration

	configuration = Configuration()

	app = FaceAnalysisWrapper(
		name=configuration.face_analysis_model_name,
		root=str(configuration.face_analysis_model_directory_path))

	face_information = app.get_face_info_from_image(
		configuration.face_image_path)
	pose_information = app.get_face_info_from_image(
		configuration.pose_image_path)

	return configuration, face_information, pose_information

def get_user_input(input_type, message, default_value=None):
	"""
	General purpose function to get user input with type validation.
	"""
	while True:
		user_input = input(
			f"{message} [{default_value if default_value is not None else 'Required'}]: ")
		if not user_input and default is not None:
			return default_value
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

def display_and_save_image(image, temp_dir):
	"""
	Display the image and provide an option to save it.
	"""
	# Display the image using the default image viewer.
	image.show()

	save_image = get_user_input(str, "Do you want to save this image? (yes/no)", "no")

	if save_image.lower() == "yes":
		file_name = get_user_input(
			str,
			"Enter filename to save image (without extension)")
		# Ensure user entered a name.
		if file_name:
			file_path = Path(temp_dir) / file_name
			image.save(file_path)
			print(f"Image saved to {file_path}")
		else:
			print("No valid file name provided; image not saved.")
	else:
		print("Iamge not saved.")

def terminal_only_main():

	FloatParameter = namedtuple('FloatParameter', ['value'])
	IntParameter = namedtuple('IntParameter', ['value'])
	StringParameter = namedtuple('StringParameter', ['value'])

	configuration, face_information, pose_information = \
		initialize_terminal_only_main()

	controlnet = create_controlnet(configuration.control_net_model_path)
	pipe = create_stable_diffusion_xl_pipeline(
		controlnet,
		configuration.diffusion_model_path,
		configuration.ip_adapter_path)

	prompt = StringParameter(get_user_input(str, "Prompt: "))
	# Example negative prompt:
	# "(lowres, low quality, worst quality:1.2), (text:1.2), glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
	# prompt for what you want to not include.
	negative_prompt = StringParameter(get_user_input(str, "Negative prompt: "))

	ip_adapter_scale = FloatParameter(
		get_user_input(
			float,
			"IP Adapter Scale: Enter float value from 0 to 1.5, normally 0.8"))

	controlnet_conditioning_scale = FloatParameter(
		get_user_input(
			float,
			"ControlNet Conditioning: Enter float value normally 0.8 or 1.0"))

	number_of_steps = IntParameter(
		get_user_input(int, "Number of steps, normally 50"))

	while True:

		image = generate_image(
			pipe,
			prompt=prompt.value,
			face_information=face_information,
			negative_prompt=negative_prompt.value,
			pose_information=pose_information,
			ip_adapter_scale=ip_adapter_scale.value,
			controlnet_conditioning_scale=controlnet_conditioning_scale.value,
			number_of_steps=number_of_steps.value)