from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class JASCOFilePathConfiguration:
    """Configuration for file paths used by JASCO wrapper"""
    # Input paths
    jasco_configuration_path: Path
    jasco_generation_configuration_path: Path
    chords_mapping_path: Path
    
    # Optional input paths
    melody_path: Optional[Path] = None
    drums_path: Optional[Path] = None
    
    # Output path
    output_directory: Path = field(default_factory=lambda: Path.cwd())
    output_prefix: str = "jasco_output"
    
    @classmethod
    def from_yaml(cls, yaml_path: Path):
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert string paths to Path objects
        for key, value in config_dict.items():
            if key.endswith('_path') or key == 'output_directory':
                if value is not None:
                    config_dict[key] = Path(value)
        
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: Path):
        import yaml
        
        # Convert Path objects to strings
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
