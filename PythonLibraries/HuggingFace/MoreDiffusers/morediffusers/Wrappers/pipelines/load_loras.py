def load_loras(pipe, loras_configuration):
    """
    @param loras_configuration LoRAsConfiguration class
    """
    if loras_configuration.lora_scale != None:
        pipe._lora_scale = loras_configuration.lora_scale
        print(f"Given lora_scale {loras_configuration.lora_scale}, " + \
            "set pipe's lora_scale to {pipe._lora_scale}")

    if len(loras_configuration.loras) == 0:
        return

    adapter_names = []
    adapter_weights = []

    return_load_state_dicts = []

    for key, lora_parameters in loras_configuration.loras.items():

        # This depends upon the interface or function signature of the class
        # methods to not change; in particular, monitor in diffusers
        # load_pipeline.py StableDiffusionXLPipeline and FluxLoraLoaderMixin.
        pipe.load_lora_weights(
            lora_parameters["directory_path"],
            weight_name=lora_parameters["weight_name"],
            adapter_name=lora_parameters["adapter_name"])

        adapter_names.append(lora_parameters["adapter_name"])
        adapter_weights.append(lora_parameters["adapter_weight"])

        try:
            return_load_state_dicts.append(pipe.lora_state_dict(
                lora_parameters["directory_path"] + \
                    "/" + \
                    lora_parameters['weight_name']))
        except Exception as err:
            print(f"Couldn't load_state_dict for this LoRA class: {err}")
            pass

    if len(loras_configuration.loras) > 1:

        # This is found in diffusers lora_base.py LorabaseMixin class, which
        # StableDiffusionXLLoraLoaderMixin and FluxLoraLoaderMixin inherit from.
        pipe.set_adapters(adapter_names, adapter_weights)

def change_pipe_with_loras_to_cuda_or_not(pipe, loras_configuration):
    if loras_configuration.is_to_cuda == True:
        pipe.to("cuda")