from .LoRALogEntry import LoRALogEntry
from .NunchakuGenerationLogEntry import NunchakuGenerationLogEntry

from morediffusers.Configurations import (
    NunchakuConfiguration,
    FluxGenerationConfiguration,
    PipelineInputs,
    NunchakuLoRAsConfiguration)

from pathlib import Path
from typing import List, Optional

import yaml

class NunchakuGenerationLogger:
    """Manages logging of Nunchaku generation parameters."""
    
    def __init__(self, log_file_path: Path):
        self.log_file_path = log_file_path
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[NunchakuGenerationLogEntry] = []

        # Create empty log file if it doesn't exist
        if not self.log_file_path.exists():
            self._initialize_empty_log_file()
   
        self._load_existing_logs()

    def _initialize_empty_log_file(self) -> None:
        """Create an empty log file with the correct YAML structure."""
        initial_data = {'generations': []}
        
        with self.log_file_path.open('w') as f:
            yaml.dump(
                initial_data,
                f,
                default_flow_style=False,
                indent=2,
                allow_unicode=True,
                # Prevents line wrapper; remove this line for default 100
                # character line limit.
                width=float('inf')
            )

    def _load_existing_logs(self) -> None:
        if self.log_file_path.exists():
            try:
                with self.log_file_path.open('r') as f:
                    data = yaml.safe_load(f) or {}
                    entries_data = data.get('generations', [])
                    self._entries = [
                        NunchakuGenerationLogEntry(**entry) 
                        for entry in entries_data
                    ]
            except Exception as e:
                # If file is corrupted, start fresh
                self._entries = []
    
    def log_generation(
        self,
        nunchaku_config: NunchakuConfiguration,
        flux_gen_config: FluxGenerationConfiguration,
        pipeline_inputs: PipelineInputs,
        loras_config: NunchakuLoRAsConfiguration,
        nunchaku_model_index: int = 0,
        generation_hash: Optional[str] = None,
        truncated_generation_hash: Optional[str] = None,
    ) -> None:
        """Log a generation with all parameters."""
        
        # Extract nunchaku model info
        nunchaku_path = Path(
            nunchaku_config.nunchaku_model_paths[nunchaku_model_index]
        )

        # Determine if path is a file or directory
        if nunchaku_path.is_file():
            # It's a file: parent dir name is the parent directory's name, filename is the file name
            nunchaku_model_parent_dir = nunchaku_path.parent.name
            nunchaku_model_filename = nunchaku_path.name
        elif nunchaku_path.is_dir():
            # It's a directory: use the directory name itself, filename is None
            nunchaku_model_parent_dir = nunchaku_path.name
            nunchaku_model_filename = None
        else:
            # Path doesn't exist yet, but we can still extract from the path
            # structure. Check if it looks like a file (has an extension) or
            # directory
            if nunchaku_path.suffix:
                # Has extension, treat as file
                nunchaku_model_parent_dir = nunchaku_path.parent.name
                nunchaku_model_filename = nunchaku_path.name
            else:
                # No extension, treat as directory
                nunchaku_model_parent_dir = nunchaku_path.name
                nunchaku_model_filename = None
            
        # Extract active LoRAs
        active_loras = loras_config.get_active_loras()
        lora_entries = [
            LoRALogEntry(
                nickname=nickname,
                lora_strength=params.lora_strength,
                filename=params.filename
            )
            for nickname, params in active_loras.items()
        ]
        
        # Create log entry
        entry = NunchakuGenerationLogEntry(
            nunchaku_model_parent_dir=nunchaku_model_parent_dir,
            nunchaku_model_filename=nunchaku_model_filename,
            height=flux_gen_config.height,
            width=flux_gen_config.width,
            num_inference_steps=flux_gen_config.num_inference_steps,
            seed=flux_gen_config.seed,
            true_cfg_scale=flux_gen_config.true_cfg_scale,
            guidance_scale_used=flux_gen_config.guidance_scale,  # Runtime value
            loras=lora_entries,
            prompt=pipeline_inputs.prompt,
            prompt_2=pipeline_inputs.prompt_2,
            negative_prompt=pipeline_inputs.negative_prompt,
            negative_prompt_2=pipeline_inputs.negative_prompt_2,
            generation_hash=generation_hash,
            truncated_generation_hash=truncated_generation_hash
        )
        
        self._entries.append(entry)
        self._save_logs()
    
    def _save_logs(self) -> None:
        """Save all log entries to YAML file."""
        data = {
            'generations': [entry.model_dump() for entry in self._entries]
        }
        
        with self.log_file_path.open('w') as f:
            yaml.dump(
                data, 
                f, 
                default_flow_style=False, 
                indent=2,
                allow_unicode=True,
                # Prevents line wrapper; remove this line for default 100
                # character line limit.
                width=float('inf')
            )

    def get_recent_logs(self, count: int = 10) \
        -> List[NunchakuGenerationLogEntry]:
        return self._entries[-count:]