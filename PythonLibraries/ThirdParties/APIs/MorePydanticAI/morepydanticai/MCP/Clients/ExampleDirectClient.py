from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pathlib import Path
import os

class ExampleDirectClient:
    def __init__(self, path_to_mcp_server: str):
        path_to_mcp_server = Path(path_to_mcp_server)
        if not path_to_mcp_server.exists():
            raise FileNotFoundError(
                f"MCP server not found at {path_to_mcp_server}")
        self.path_to_mcp_server = path_to_mcp_server

    async def client(self):
        server_parameters = StdioServerParameters(
#            command='uv',
#            args=['run', str(self.path_to_mcp_server), 'server'],
#            env=os.environ
            command='python',
            args=[str(self.path_to_mcp_server)],
            env=os.environ
        )

        async with stdio_client(server_parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool('poet', {'theme': 'socks'})
                print(result.content[0].text)

    async def general_client(self, theme: str):
        server_parameters = StdioServerParameters(
            command='python',
            args=[str(self.path_to_mcp_server)],
            env=os.environ
        )

        async with stdio_client(server_parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool('poet', {'theme': theme})
                print(result.content[0].text)