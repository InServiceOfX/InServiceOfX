def load_ip_adapter(pipe, ip_adapter_configuration):

	# See diffusers/src/diffusers/loaders/ip_adapter.py for class
	# IPAdapterMixin, load_ip_adapter method.

	pipe.load_ip_adapter(
		# pretrained_model_name_or_path_or_dict
		ip_adapter_configuration.path,
		# subfolder
		ip_adapter_configuration.subfolder,
		# weight_name, str or List[str]
		ip_adapter_configuration.weight_names,
		# image_encoder_folder (optional)
		local_files_only=True,
		)

	pipe.set_ip_adapter_scale(ip_adapter_configuration.scales)