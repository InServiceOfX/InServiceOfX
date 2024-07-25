from diffusers import DDIMScheduler

def create_ddim_scheduler(
    diffusion_model_subdirectory,
    subfolder=None,
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1):
    """
    See diffusers, schedulers.scheduling_ddim.py class DDIMScheduler(
    SchedulerMixin, ConfigMixin) for class definition and
    schedulers/scheduling_utils.py for SchedulerMixin for
    def from_pretrained(..) definition.

    .from_pretrained(..) from scheduling_utils.py calls load_config of
    configuration_utils.py which calls register_to_config(..) and
    register_to_config also is a "prefix" to def __init__(..) in
    scheduling_ddim.py.

    @param diffusion_model_subdirectory: pathlib.Path
        This is pretrained_model_name_or_path in def from_pretrained(..)
    @param subfolder (str, optional)
        This is subfolder location of model file within larger model repository.
    @param clip_sample (bool,)
        Originally, this is clip_sample in DDIMScheduler.__init__(..) and
        originally defaults to True. Clip predicted sample for numerical
        stability.
    @param timestep_spacing (str,)
        Originally, timestep_spacing in DDIMScheduler.__init__(..), the way
        timesteps should be scaled. Refer to Table 2 of
        [Common Diffusion Noise Schedules and Sample Steps are Flawed]
        (https://huggingface.co/papers/2305.08891).
        Originally defaults to "leading"
    @param beta_schedule (str, defaults to "linear")
        The beta schedule, a mapping from a beta range to a sequence of betas
        for stepping the model. Choose
        linear
        scaled_linear
        squaredcos_cap_v2
    @param steps_offset (int, )
        Originally in DDIMScheduler.__init__(..) as steps_offset, an offset
        added to inference steps, as required by some model families.
        Originally defaults to 0.
    """
    scheduler = DDIMScheduler.from_pretrained(
        diffusion_model_subdirectory,
        subfolder=subfolder,
        clip_sample=clip_sample,
        timestep_spacing=timestep_spacing,
        beta_schedule=beta_schedule,
        steps_offset=steps_offset,
        local_files_only=True)

    return scheduler
