from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path
import os

class ExampleMultiToolClient:
    def __init__(self, path_to_mcp_server: str):
        path_to_mcp_server = Path(path_to_mcp_server)
        if not path_to_mcp_server.exists():
            raise FileNotFoundError(
                f"MCP server not found at {path_to_mcp_server}")
        self.path_to_mcp_server = path_to_mcp_server

    async def run_all_poems(self, theme: str):
        """Run all poem types for a given theme"""
        server_parameters = StdioServerParameters(
            command='python',
            args=[str(self.path_to_mcp_server)],
            env=os.environ
        )

        async with stdio_client(server_parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Get all available tools
                tools = await session.list_tools()
                print(f"Available tools: {[tool.name for tool in tools.tools]}")
                
                # Run each poem type
                for poem_type in ['poet', 'haiku', 'limerick']:
                    print(f"\n=== {poem_type.upper()} ===")
                    result = await session.call_tool(poem_type, {'theme': theme})
                    print(result.content[0].text)

