from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import socket
import yaml

@dataclass
class ServerConfiguration:
    model_path: Union[str, Path]
    mem_fraction_static: Optional[float] = None
    port: Optional[int] = None
    host: Optional[str] = None
    
    def has_server_fields(self) -> bool:
        """Check if configuration has required server fields."""
        return all([
            self.model_path is not None,
            self.port is not None,
            self.host is not None
        ])
    
    def set_to_available_defaults(self) -> None:
        """Set default host and find available port."""
        self.host = "0.0.0.0"
        
        # Try ports from 30000 to 30022
        for test_port in range(30000, 30023):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((self.host, test_port))
                self.port = test_port
                sock.close()
                return
            except socket.error:
                sock.close()
                continue
                
        raise RuntimeError(
            "No available ports found in range 30000-30022"
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ServerConfiguration':
        """Load configuration from yaml file."""
        with open(yaml_path, 'r') as file:
            config_dict = yaml.safe_load(file)
            
        if not config_dict:
            raise ValueError("YAML file is empty")
            
        if 'model_path' not in config_dict:
            raise ValueError("model_path is required in configuration")
            
        # Convert model_path to Path if it's a string
        if isinstance(config_dict.get('model_path'), str):
            config_dict['model_path'] = Path(config_dict['model_path'])
            
        # Only pass keys that exist in the yaml and match our dataclass fields
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {
            k: v for k, v in config_dict.items() 
            if k in valid_keys and v is not None
        }
        
        return cls(**filtered_dict)


