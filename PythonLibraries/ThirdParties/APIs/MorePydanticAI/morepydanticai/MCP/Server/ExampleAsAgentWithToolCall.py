from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent

class ExampleAsAgentWithToolCall:
    def __init__(self):
        self.mcp_server = FastMCP("Pydantic AI Server")
        self.server_agent = Agent(
            "groq:llama-3.1-8b-instant",
            system_prompt='always reply in rhyme')

    @self.mcp_server.tool()
    async def poet(self, theme: str) -> str:
        """Poem generator"""
        r = await self.server_agent.run(f'write a poem about {theme}')
        return r.output

if __name__ == "__main__":
    server = ExampleAsAgentWithToolCall()
    server.mcp_server.run()