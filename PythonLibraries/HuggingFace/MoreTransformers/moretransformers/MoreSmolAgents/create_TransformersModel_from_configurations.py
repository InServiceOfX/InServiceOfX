from smolagents import TransformersModel

def create_TransformersModel_from_configurations(
        from_pretrained_model_configuration,
        generation_configuration):
    return TransformersModel(
        model_id=str(
            from_pretrained_model_configuration.pretrained_model_name_or_path),
        device_map=from_pretrained_model_configuration.device_map,
        torch_dtype=from_pretrained_model_configuration.torch_dtype,
        trust_remote_code=from_pretrained_model_configuration.trust_remote_code,
        **generation_configuration.to_dict())
