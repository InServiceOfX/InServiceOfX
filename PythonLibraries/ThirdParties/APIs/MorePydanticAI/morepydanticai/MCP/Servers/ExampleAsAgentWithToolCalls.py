from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent

class ExampleAsAgentWithToolCalls:

    SYSTEM_PROMPT = """always reply in rhyme.
    
    You are a creative, talented, adroit, and versatile poet.
    """


    def __init__(self):
        self.mcp_server = FastMCP("Pydantic AI Server")
        self.server_agent = Agent(
            "groq:llama-3.1-8b-instant",
            system_prompt=self.SYSTEM_PROMPT)
        
        # Register all tools after instance creation
        self.poet = self.mcp_server.tool()(self.poet)
        self.haiku = self.mcp_server.tool()(self.haiku)
        self.limerick = self.mcp_server.tool()(self.limerick)

    # Originally, we needed to decorate the tool like so
    # @mcp_server.tool()
    # async def poet(..)
    # But self and the instance wouldn't have been created yet.

    async def poet(self, theme: str) -> str:
        """Generate a poem about a given theme"""
        r = await self.server_agent.run(f'write a poem about {theme}')
        return r.output

    async def haiku(self, theme: str) -> str:
        """Generate a haiku about a given theme"""
        r = await self.server_agent.run(f'write a haiku about {theme}')
        return r.output

    async def limerick(self, theme: str) -> str:
        """Generate a limerick about a given theme"""
        r = await self.server_agent.run(f'write a limerick about {theme}')
        return r.output

if __name__ == "__main__":
    server = ExampleAsAgentWithToolCalls()
    server.mcp_server.run()