from textwrap import dedent

class SimpleZeroShotRAGAgent:
    """
    Simple zero-shot RAG agent using system prompt for routing.
    Decides via keywords if to call retrieve_context for past conversations.
    Best practice: Prompt-based routing for low-complexity.
    """
    
    def __init__(
        self,
        model_manager: ModelAndToolCallManager,
        rag_tool: RAGTool,
        system_prompt_template: str = None
    ):
        self.manager = model_manager
        self.rag_tool = rag_tool
        self.processor = ToolCallProcessor()  # From your codebase
        self._register_tool()
        
        # Default system prompt (customize; based on OpenAI/LangChain best practices)
        self.system_prompt_template = system_prompt_template or self._default_system_prompt()
    
    @staticmethod
    def _default_system_prompt() -> str:
        """Prompt engineering: Explicit instructions + tool schema."""
        return dedent("""
You are a helpful assistant with access to past conversation history via a RAG tool.

TOOL: retrieve_context
- Description: Retrieves relevant snippets from previous conversations in the vector database.
- When to use: ONLY if the user explicitly wants to recall or search past info. Look for triggers like:
  - "Recall" or "Remember"
  - "Search previous/past conversation(s)"
  - "Look up earlier discussion"
  - Similar phrases indicating memory lookup.
- Input: A query string summarizing what to recall/search (e.g., "our chat about Python testing").
- Output: Use the returned context to inform your response.

If no triggers, respond directly using your knowledge. Do not hallucinate history.
Always be concise and helpful.
""")