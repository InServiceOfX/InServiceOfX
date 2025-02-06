from dataclasses import dataclass
import atexit
import time
from pathlib import Path
from typing import Optional
from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process
)

from moresglang.Configurations import ServerConfiguration

@dataclass
class HTTPServer:
    config: ServerConfiguration
    server_process: Optional[object] = None
    
    def __post_init__(self):
        # Register cleanup on program exit
        atexit.register(self.shutdown)
    
    def start(self):
        """Start the server with configuration."""
        if not self.config.has_server_fields():
            self.config.set_to_available_defaults()
            
        cmd = f"""python -m sglang.launch_server \
            --model-path {self.config.model_path} \
            --port {self.config.port} \
            --host {self.config.host}"""
            
        if self.config.mem_fraction_static is not None:
            cmd += f" --mem-fraction-static {self.config.mem_fraction_static}"
            
        self.server_process = execute_shell_command(cmd)
        
        # Wait for server is necessary to ensure server is ready
        wait_for_server(f"http://localhost:{self.config.port}")
        
    def shutdown(self):
        """Cleanup server process."""
        if self.server_process:
            terminate_process(self.server_process)
            self.server_process = None
            time.sleep(2)  # Allow time for cleanup
            
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.shutdown()
