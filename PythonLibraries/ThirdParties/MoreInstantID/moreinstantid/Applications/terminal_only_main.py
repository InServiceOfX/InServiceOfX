from collections import namedtuple
from pathlib import Path
import sys
import time

python_libraries_path = Path(__file__).resolve().parent.parent.parent.parent.parent
corecode_directory = python_libraries_path / "CoreCode"
more_insightface_directory = \
	python_libraries_path / "ThirdParties" / "MoreInsightFace"
more_instant_id_directory = \
	python_libraries_path / "ThirdParties" / "MoreInstantID"

instant_id_directory = python_libraries_path.parent.parent / "ThirdParty" / \
	"InstantID"

if not str(corecode_directory) in sys.path:
	sys.path.append(str(corecode_directory))
if not str(more_insightface_directory) in sys.path:
	sys.path.append(str(more_insightface_directory))
if not str(more_instant_id_directory) in sys.path:
	sys.path.append(str(more_instant_id_directory))
if not str(instant_id_directory) in sys.path:
	sys.path.append(str(instant_id_directory))

from corecode.Utilities import (
	clear_torch_cache_and_collect_garbage,
	get_user_input,
	FloatParameter,
	IntParameter,
	StringParameter)
from moreinsightface.Wrappers import get_face_and_pose_info_from_images
from moreinstantid.Wrappers import (
	create_controlnet,
	create_stable_diffusion_xl_pipeline,
	generate_image)
from moreinstantid.Configuration import Configuration


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
			# Determine image format and add the appropriate extension.
			image_format = image.format if image.format else "PNG"
			file_path = Path(temp_dir) / f"{file_name}.{image_format.lower()}"
			image.save(file_path)
			print(f"Image saved to {file_path}")
		else:
			print("No valid file name provided; image not saved.")
	else:
		print("Image not saved.")

def terminal_only_main():

	start_time = time.time()

	configuration = Configuration()

    face_information, pose_information = get_face_and_pose_info_from_images(
        model_name=configuration.face_analysis_model_name,
        model_root_directory=str(
            configuration.face_analysis_model_directory_path),
        face_image_path=configuration.face_image_path,
        pose_image_path=configuration.pose_image_path)

	end_time = time.time()
	duration = end_time - start_time

	print("-------------------------------------------------------------------")
	print("Completed initialization, obtained configuration, face, post information.")
	print(f"Took {duration:.2f} seconds to initialize.")
	print("-------------------------------------------------------------------")

	start_time = time.time()

	# The ControlNet binary offered by InstantID doesn't seem to work with
	# 16-bit float type from torch. So we didn't add the torch_dtype argument.
	controlnet = create_controlnet(configuration.control_net_model_path)
	pipe = create_stable_diffusion_xl_pipeline(
		controlnet,
		configuration.diffusion_model_path,
		configuration.ip_adapter_path,
		is_enable_cpu_offload=True,
		is_enable_sequential_cpu=True)

	end_time = time.time()
	duration = end_time - start_time

	print("-------------------------------------------------------------------")
	print(f"Completed pipeline creation, took {duration:.2f} seconds.")
	print("-------------------------------------------------------------------")

	prompt = StringParameter(get_user_input(str, "Prompt: "))
	# Example negative prompt:
	# "(lowres, low quality, worst quality:1.2), (text:1.2), glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"
	# prompt for what you want to not include.
	negative_prompt = StringParameter(get_user_input(str, "Negative prompt: ", ""))

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

	print("prompt: ", prompt.value)
	print("negative prompt: ", negative_prompt.value)
	print("IP adapter scale: ", ip_adapter_scale.value)
	print("ControlNet Conditioning: ", controlnet_conditioning_scale.value)
	print("Number of Steps: ", number_of_steps.value)

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

		display_and_save_image(image, configuration.temporary_save_path)

		# Ask if user wants to generate another image.
		continue_response = get_user_input(
			str,
			"Do you want to generate another image? (yes/no)", "no")
		if continue_response.lower() != "yes":
			break

		# Optionally update parameters

		prompt = StringParameter(
			get_user_input(
				str,
				"New Prompt or press Enter for previous: ",
				prompt.value))

		negative_prompt = StringParameter(
			get_user_input(
				str,
				"New Negative Prompt or press Enter for previous: ",
				negative_prompt.value))

		ip_adapter_scale = FloatParameter(
			get_user_input(
				float,
				"New IP Adapter Scale or press Enter for previous: ",
				ip_adapter_scale.value))

		controlnet_conditioning_scale = FloatParameter(
			get_user_input(
				float,
				"New ControlNet Conditioning Scale or press Enter for previous: ",
				controlnet_conditioning_scale.value))

		number_of_steps = IntParameter(
			get_user_input(
				int,
				"Net number of steps or press Enter for previous: ",
				number_of_steps.value))

	clear_torch_cache_and_collect_garbage()

if __name__ == "__main__":

	terminal_only_main()
