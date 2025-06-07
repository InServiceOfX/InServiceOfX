from corecode.Utilities import load_environment_file
from morepydanticai.MCP.Clients import ExampleDirectClient

from pathlib import Path

import asyncio

load_environment_file()

path_to_mcp_server = Path(__file__).parents[4] / "morepydanticai" / "MCP" / \
    "Servers" / "ExampleAsAgentWithToolCall.py"

def test_ExampleDirectClient():
    asyncio.run(ExampleDirectClient(path_to_mcp_server).client())

def test_ExampleDirectClient_general_client():
    theme = "water"
    asyncio.run(ExampleDirectClient(path_to_mcp_server).general_client(theme))