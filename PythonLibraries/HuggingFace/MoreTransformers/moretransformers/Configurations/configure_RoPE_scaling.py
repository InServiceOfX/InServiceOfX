def configure_RoPE_scaling(configuration, factor=2.0, type="yarn"):
    original_max_position_embeddings = configuration.max_position_embeddings

    configuration.rope_scaling = {
        "factor": factor,
        "original_max_position_embeddings": original_max_position_embeddings,
        "type": type
    }
    configuration.max_position_embeddings = \
        factor * original_max_position_embeddings
    return configuration