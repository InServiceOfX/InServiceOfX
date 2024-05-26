"""
@ref https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet
"""

from diffusers import (
    AmusedScheduler,
    CMStochasticIterativeScheduler,
    # TODO: for commented out classes, fix the import path.

    #ConsistencyDecoderScheduler,
    DDIMInverseScheduler,
    DDIMParallelScheduler,
    DDIMScheduler,
    DDPMParallelScheduler,
    DDPMScheduler,
    DDPMWuerstchenScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EDMEulerScheduler,
    EDMDPMSolverMultistepScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KarrasVeScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    RePaintScheduler,
    SASolverScheduler,
    ScoreSdeVeScheduler,
    #ScoreSdeVpScheduler,
    TCDScheduler,
    UniPCMultistepScheduler,
    VQDiffusionScheduler
    )

class CreateSchedulerMap:

    _scheduler_name_to_class_map = {
        "AmusedScheduler": AmusedScheduler,
        "CMStochasticIterativeScheduler": CMStochasticIterativeScheduler,
        "DDIMInverseScheduler": DDIMInverseScheduler,
        "DDIMParallelScheduler": DDIMParallelScheduler,
        "DDIMScheduler": DDIMScheduler,
        "DDPMParallelScheduler": DDPMParallelScheduler,
        "DDPMScheduler": DDPMParallelScheduler,
        "DDPMWuerstchenScheduler": DDPMWuerstchenScheduler,
        "DEISMultistepScheduler": DEISMultistepScheduler,
        "DPMSolverMultistepInverseScheduler": DPMSolverMultistepInverseScheduler,
        "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "PNDMScheduler": PNDMScheduler,
        "SASolverScheduler": SASolverScheduler,
        "ScoreSdeVeScheduler": ScoreSdeVeScheduler,
        "TCDScheduler": TCDScheduler,
        "UniPCMultistepScheduler": UniPCMultistepScheduler,
        "VQDiffusionScheduler": VQDiffusionScheduler
    }

    @classmethod
    def get_map(cls):
        return cls._scheduler_name_to_class_map