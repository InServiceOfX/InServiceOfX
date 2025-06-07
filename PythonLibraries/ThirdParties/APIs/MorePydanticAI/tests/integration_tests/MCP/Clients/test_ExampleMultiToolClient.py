from corecode.Utilities import load_environment_file
from morepydanticai.MCP.Clients import ExampleMultiToolClient

from pathlib import Path

import asyncio

load_environment_file()

path_to_mcp_server = Path(__file__).parents[4] / "morepydanticai" / "MCP" / \
    "Servers" / "ExampleAsAgentWithToolCalls.py"

def test_ExampleMultiToolClient():
    asyncio.run(ExampleMultiToolClient(
        path_to_mcp_server).run_all_poems('space'))
