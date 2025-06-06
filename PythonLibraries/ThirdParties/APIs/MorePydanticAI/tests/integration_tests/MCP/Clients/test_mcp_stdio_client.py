from corecode.Utilities import (load_environment_file)

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

import pytest

load_environment_file()

@pytest.mark.asyncio
async def test_mcp_stdio_client():

    # Use the IP address of the running MCP server. If the MCP server is running
    # locally on a host computer, get the IP address of the host computer.
    server = MCPServerStdio(
        'deno',
        args=[
            'run',
            '-N',
            '-R=node_modules',
            '-W=node_modules',
            '--node-modules-dir=auto',
            'jsr:@pydantic/mcp-run-python',
            'stdio',
        ],)

    agent = Agent("groq:llama-3.1-8b-instant", mcp_servers=[server,])

    async with agent.run_mcp_servers():
        result = await agent.run(
            "How many days between 2000-01-01 and 2025-03-18?"
        )
    print(result.output)

    assert "9208 days" in result.output

def test_tool_prefixes_to_avoid_naming_conflicts():
    python_server = MCPServerStdio(
        'deno',
        args=[
            'run',
            '-N',
            'jsr:@pydantic/mcp-run-python',
            'stdio'],
        # Tools will be prefixed with 'py_'
        tool_prefix='py')

    js_server = MCPServerStdio(
        'deno',
        args=[
            'run',
            '-N',
            'jsr:@pydantic/mcp-run-python',
            'stdio'],
        # Tools will be prefixed with 'js_'
        tool_prefix='js')

    agent = Agent(
        "groq:llama-3.1-8b-instant",
        mcp_servers=[python_server, js_server])

