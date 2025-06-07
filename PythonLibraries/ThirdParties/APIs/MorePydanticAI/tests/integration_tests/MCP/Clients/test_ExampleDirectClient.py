from morepydanticai.MCP.Clients import ExampleDirectClient

from pathlib import Path

import asyncio

path_to_mcp_server = Path(__file__).parent / "ExampleServer"

def test_ExampleDirectClient():
    asyncio.run(ExampleDirectClient(path_to_mcp_server).client())