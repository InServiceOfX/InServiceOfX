from corecode.FileIO import is_directory_empty_or_missing
from corecode.Utilities import DataSubdirectories
from moresglang.Configurations import ServerConfiguration
from moresglang.Wrappers.EntryPoints import HTTPServer
from pathlib import Path
# TODO: See if this is needed.
#from sglang.utils import print_highlight

from openai.types.chat.chat_completion import ChatCompletion

import openai
import pytest

test_data_directory = Path(__file__).resolve().parents[3] / "TestData"
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
    config = ServerConfiguration.from_yaml(
        test_data_directory / "server_configuration.yml")
    config.model_path = MODEL_DIR

    server = HTTPServer(config)
    server.start()

    assert server.server_process is not None
    assert server.server_process.poll() is None

    server.shutdown()

@pytest.mark.skipif(
    is_directory_empty_or_missing(MODEL_DIR),
    reason=skip_reason
)
def test_server_handles_openai_client_request():
    # Setup server
    config = ServerConfiguration.from_yaml(
        test_data_directory / "server_configuration.yml")
    config.model_path = MODEL_DIR
    
    server = HTTPServer(config)
    server.start()
    
    try:
        # Setup OpenAI client
        client = openai.Client(
            base_url=f"http://localhost:{config.port}/v1",
            api_key="None"
        )
        
        # Make request
        response = client.chat.completions.create(
            model=str(MODEL_DIR.name),
            messages=[
                {"role": "user", "content": "What is the capital of France?"}
            ],
            temperature=0,
            max_tokens=256,
        )

        print(response)

        assert isinstance(response, ChatCompletion)

        # Assertions
        assert response is not None
        assert len(response.choices) > 0
        print(len(response.choices))
        assert response.choices[0].message.content is not None
       
        # TODO: See what print_highlight does.
        #print_highlight(f"Response: {response}")
        
        # TODO: Clean this up.
        # Print all available attributes
        #print("\nChatCompletion attributes:")
        #for attr in dir(response):
            # Skip private attributes
            #if not attr.startswith('_'):
                #print(f"{attr}: {getattr(response, attr)}")
                
        # Common attributes you can access:
        assert response.id  # Unique identifier
        assert response.model == "DeepSeek-R1-Distill-Qwen-1.5B"
        assert response.model_extra == {}
        assert response.object == "chat.completion"
        assert response.created  # Timestamp
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content  # The actual response
        assert response.usage.completion_tokens  # Token usage stats
        assert response.usage.prompt_tokens
        assert response.usage.total_tokens
        assert response.service_tier == None
        assert response.system_fingerprint == None

    finally:
        server.shutdown()
