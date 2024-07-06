from ..Schedulers import CreateSchedulerMap

def change_scheduler_or_not(pipe, scheduler_name=None, a1111_kdiffusion=None):
    if scheduler_name == None:
        return False
    schedulers_map = CreateSchedulerMap.get_map()
    pipe.scheduler = schedulers_map[scheduler_name].from_config(
        pipe.scheduler.config)

    # https://huggingface.co/stabilityai/stable-diffusion-2-1/discussions/23
    # so DPM++2M Karras is DPMSolverMultistepScheduler but with
    # init with use_karras_sigmas=True
    # https://huggingface.co/docs/diffusers/v0.26.2/en/api/schedulers/overview#schedulers

    if a1111_kdiffusion != None:

        if scheduler_name == "DPMSolverMultistepScheduler":

            if a1111_kdiffusion == "DPM++ 2M Karras":
                pipe.scheduler.use_karras_sigmas=True
            elif a1111_kdiffusion == "DPM++ 2M SDE":
                pipe.scheduler.config.algorithm_type = "sde-dpmsolver++"

        elif scheduler_name == "DPMSolverSinglestepScheduler":
            if a1111_kdiffusion == "DPM++ SDE Karras":
                pipe.scheduler.use_karras_sigmas=True

    return True