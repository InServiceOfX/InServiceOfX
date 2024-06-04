def load_loras(
	pipe,
	loras_configuration
    ):
    """
    @param loras_configuration LoRAsConfiguration class
    """
    if len(loras_configuration.loras) == 0:
        return

    adapter_names = []
    adapter_weights = []

    for key, lora_parameters in loras_configuration.loras.items():

        pipe.load_lora_weights(
            lora_parameters["directory_path"],
            weight_name=lora_parameters["weight_name"],
            adapter_name=lora_parameters["adapter_name"])

        adapter_names.append(lora_parameters["adapter_name"])
        adapter_weights.append(lora_parameters["adapter_weight"])

    if len(loras_configuration.loras) > 1:

        pipe.set_adapters(adapter_names, adapter_weights)

