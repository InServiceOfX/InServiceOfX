from ..Schedulers import CreateSchedulerMap
from typing import Optional

KDIFFUSION_CONFIGS = {
    "DPMSolverMultistepScheduler": {
        "DPM++ 2M Karras": [("use_karras_sigmas", True)],
        "DPM++ 2M SDE": [("config.algorithm_type", "sde-dpmsolver++")]
    },
    "DPMSolverSinglestepScheduler": {
        "DPM++ SDE Karras": [("use_karras_sigmas", True)]
    }
}

def change_scheduler_or_not(
    pipe,
    scheduler_name: Optional[str] = None,
    a1111_kdiffusion: Optional[str] = None) -> bool:
    if scheduler_name is None:
        return False
        
    schedulers_map = CreateSchedulerMap.get_map()
    pipe.scheduler = schedulers_map[scheduler_name].from_config(
        pipe.scheduler.config)

    # https://huggingface.co/stabilityai/stable-diffusion-2-1/discussions/23
    # so DPM++2M Karras is DPMSolverMultistepScheduler but with
    # init with use_karras_sigmas=True
    # https://huggingface.co/docs/diffusers/v0.26.2/en/api/schedulers/overview#schedulers

    if a1111_kdiffusion is not None:
        if (scheduler_name in KDIFFUSION_CONFIGS and 
            a1111_kdiffusion in KDIFFUSION_CONFIGS[scheduler_name]):
            
            for attr_path, value in \
                KDIFFUSION_CONFIGS[scheduler_name][a1111_kdiffusion]:
                obj = pipe.scheduler
                *path_parts, final_attr = attr_path.split('.')
                
                # Navigate nested attributes
                for part in path_parts:
                    obj = getattr(obj, part)
                
                # Set final attribute
                setattr(obj, final_attr, value)

    return True