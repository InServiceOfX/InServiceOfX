def change_pipe_to_cuda_or_not(configuration, pipe):
    if configuration.is_to_cuda is True:
        to_kwargs = {
            "device": getattr(configuration, "cuda_device", "cuda")}
        if (configuration.torch_dtype is not None):
            to_kwargs["dtype"] = configuration.torch_dtype
        pipe.to(**to_kwargs)
