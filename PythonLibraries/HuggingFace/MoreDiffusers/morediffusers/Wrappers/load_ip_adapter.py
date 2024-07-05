def load_ip_adapter(pipe, ip_adapter_configuration, is_load_image_encoder=True):
	"""
	See diffusers/src/diffusers/loaders/ip_adapter.py for class
	IPAdapterMixin, load_ip_adapter method.

	Recall that image_encoder_folder:
	The subfolder location of the image encoder within a larger model
	repository. Pass 'None' to not load the image encoder.
	"""

	if is_load_image_encoder:
		pipe.load_ip_adapter(
			# pretrained_model_name_or_path_or_dict
			ip_adapter_configuration.path,
			# subfolder
			ip_adapter_configuration.subfolder,
			# weight_name, str or List[str]
			ip_adapter_configuration.weight_names,
			# image_encoder_folder (optional) where Optional[str] = "image_encoder"
			# so the default value is str "image_encoder".
			local_files_only=True,
			)
	else:
		pipe.load_ip_adapter(
			# pretrained_model_name_or_path_or_dict
			ip_adapter_configuration.path,
			# subfolder
			ip_adapter_configuration.subfolder,
			# weight_name, str or List[str]
			ip_adapter_configuration.weight_names,
			image_encoder_folder=None,
			local_files_only=True,
			)		

	pipe.set_ip_adapter_scale(ip_adapter_configuration.scales)