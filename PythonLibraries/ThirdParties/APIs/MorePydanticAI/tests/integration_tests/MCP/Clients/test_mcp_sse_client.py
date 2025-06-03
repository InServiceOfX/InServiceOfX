"""
In another terminal,

deno run \
  -N -R=node_modules -W=node_modules --node-modules-dir=auto \
  jsr:@pydantic/mcp-run-python sse
"""
from corecode.Utilities import (load_environment_file)

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

import pytest

load_environment_file()

@pytest.mark.asyncio
async def test_mcp_sse_client():
    server = MCPServerHTTP(url='http://localhost:3001/sse')
    agent = Agent("groq:llama-3.1-8b-instant")

    async with agent.run_mcp_servers():
        result = await agent.run(
            "How many days between 2000-01-01 and 2025-03-18?"
        )
    print(result.output)