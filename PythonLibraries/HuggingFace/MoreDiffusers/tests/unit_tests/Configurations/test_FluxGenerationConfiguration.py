from morediffusers.Configurations import FluxGenerationConfiguration

def test_FluxGenerationConfiguration_inits_for_empty_values():
    configuration = FluxGenerationConfiguration()

    assert configuration.true_cfg_scale is None
    assert configuration.height is None
    assert configuration.width is None
    assert configuration.num_inference_steps is None
    assert configuration.num_images_per_prompt is None
    assert configuration.seed is None
    assert configuration.guidance_scale is None
    assert configuration.max_sequence_length is None
