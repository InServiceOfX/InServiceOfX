"""
In another terminal,

deno run \
  -N -R=node_modules -W=node_modules --node-modules-dir=auto \
  jsr:@pydantic/mcp-run-python sse
"""
from corecode.Utilities import load_environment_file

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

import pytest

load_environment_file()

# Use the IP address of the running MCP server. If the MCP server is running
# locally on a host computer, get the IP address of the host computer.
IP_ADDRESS = 'XXX.XXX.XX.XX'

@pytest.mark.asyncio
async def test_mcp_sse_client():

    server = MCPServerHTTP(url=f'http://{IP_ADDRESS}:3001/sse')
    agent = Agent("groq:llama-3.1-8b-instant", mcp_servers=[server,])

    async with agent.run_mcp_servers():
        result = await agent.run(
            "How many days between 2000-01-01 and 2025-03-18?"
        )
    print(result.output)

    assert "9208" in result.output

def test_tool_prefixes_to_avoid_naming_conflicts():
    # Create two servers with different prefixes.
    weather_server = MCPServerHTTP(
        url=f'http://{IP_ADDRESS}:3001/sse',
        # Tools will be prefixed with 'weather_'
        tool_prefix='weather')

    calculator_server = MCPServerHTTP(
        url=f'http://{IP_ADDRESS}:3001/sse',
        # Tools will be prefixed with 'calc_'
        tool_prefix='calc')

    # Both servers might have a tool named 'get_data', but they'll be exposed
    # as:
    # - 'weather_get_data'
    # - 'calc_get_data'

    agent = Agent(
        "groq:llama-3.1-8b-instant",
        mcp_servers=[weather_server, calculator_server])
