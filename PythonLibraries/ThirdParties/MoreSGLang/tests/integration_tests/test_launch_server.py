from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories

from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process
)

import pytest
import time

data_sub_dirs = DataSubdirectories()

MODEL_DIR = data_sub_dirs.ModelsLLM / "deepseek-ai" / \
    "DeepSeek-R1-Distill-Qwen-1.5B"

# Skip reason that will show in pytest output
skip_reason = f"Directory {MODEL_DIR} is empty or doesn't exist"

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_server_starts():
    # Start the server
    server_process = execute_shell_command(
        f"""python -m sglang.launch_server \
        --model-path {MODEL_DIR} \
        --mem-fraction-static 0.70 \
        --port 30000 \
        --host 0.0.0.0"""
    )
    
    try:
        # Wait for server to be ready
        wait_for_server("http://localhost:30000")
        
        # Add your assertions here
        assert server_process is not None
        assert server_process.poll() is None  # Check if process is running
        
    finally:
        # Cleanup: terminate server
        terminate_process(server_process)
        time.sleep(2)  # Give time for cleanup
