from transformers import pipeline
import torch

def create_pipeline(
    model_subdirectory,
    task="text-generation",
    torch_dtype=None,
    device_map="auto",
    ):
    """
    For construction, 

    modeling_utils.py, class PreTrainedModel, from_pretrained(..), device_map
    keyword argument is parsed and could be a
    torch.device,
    str and expected are
    "auto", "balanced", "balanced_low_0", "sequential"

    For wrapping Utility factory method def pipeline(..) int
    pipelines/__init__.py,
    def pipeline(
        task: str = None,
        model: Optional[Union[str, "PreTraindeModel", "TFPreTrainedModel"]] = None,
        )
    
    Given the task, e.g. "text-generation", around line 822-852,
    normalized_task, targeted_task, task_options are obtained. TODO: consider
    writing test upon using check_task of __init__.py.
    """
    return pipeline(
        task,
        model=model_subdirectory,
        torch_dtype=torch_dtype,
        device_map=device_map)