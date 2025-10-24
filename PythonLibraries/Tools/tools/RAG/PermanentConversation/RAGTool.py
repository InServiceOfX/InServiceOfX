from typing import Optional, Any, Callable, List, Dict

from .RAGProcessor import RAGProcessor

from string import Template
from textwrap import dedent

import asyncio

class RAGTool:
    """Tool that given a query, retrieves previous conversations and returns
    a new message with context."""

    CONTEXT_TEMPLATE = Template(dedent("""
        Based on the following context from previous conversations, please
        answer the user's question.
        Context:
        $context
        Question:
        $question
        Please provide a comprehensive answer based on the context provided. If
        the context doesn't contain enough information to fully answer the
        question, acknowledge this limitation while providing what information
        you can from the available context.
    """))

    DEFAULT_MAX_CHUNKS = 7
    DEFAULT_ROLE_FILTER = None
    DEFAULT_IS_RETURN_MATCHES = False

    def __init__(self, rag_processor: RAGProcessor):
        self.rag_processor = rag_processor

    async def retrieve_context(
        self,
        query: str,
        role_filter: Optional[str] = None,
        max_chunks: int = 5,
        is_return_matches: bool = False,
    ):
        """
        Retrieve context from previous conversations.
        """
        try:
            maybe = await self.rag_processor.process_query_to_context(
                query=query,
                role_filter=role_filter,
                limit=max_chunks,
                is_return_matches=is_return_matches
            )

            if is_return_matches:
                context, search_results = maybe
            else:
                context = maybe

            if not context:
                message = (
                    "No relevant conversation history found for the given "
                    "query."
                )
            else:
                message = self.CONTEXT_TEMPLATE.substitute(
                    context=context,
                    question=query
                )

            return_dict = {
                "message": message,
                "context": context,
                "search_results": None
            }

            if is_return_matches:
                return_dict["search_results"] = search_results

            return return_dict

        except Exception as err:
            error_msg = f"Error retrieving context for query: {query}: {err}"
            raise RuntimeError(error_msg)

    async def retrieve_context_as_tool(self, query: str, max_chunks=None):
        """
        Retrieve relevant context from previous conversations using RAG and
        format it for LLM augmentation.

        This function is designed as a tool-callable entrypoint for AI agents.
        It performs semantic retrieval from a vector store of past
        conversations, filters by relevance, and returns a templated string
        ready for injection into an LLM prompt.

        Best Practices Alignment:
        - Inputs minimized to query + optional limit (per LangChain custom
          tools:
          https://python.langchain.com/docs/how_to/custom_tools/).
        - Output as single str for direct LLM consumption (avoids parsing;
          aligns with OpenAI function calling:
          https://platform.openai.com/docs/guides/function-calling).
        - Verbose docstring for LLM guidance during tool selection (e.g., in
          agent routers:
          https://docs.llamaindex.ai/en/stable/module_guides/querying/router/).
        - Async for non-blocking I/O in agent loops (e.g., LlamaIndex agents:
          https://medium.com/@samad19472002/agentic-rag-application-using-llamaindex-tool-calling-30bfef6cb4fb).

        Example Usage in Agent:
            # In your SimpleRAGAgent or ModelAndToolCallManager
            tool_schema = {
                "name": "rag_retrieve_context",
                "description": "Retrieve and format previous conversation context for a query.",
                "parameters": RAGRetrieveInput.model_json_schema()  # Pydantic to JSON schema
            }
            # Bind to LLM: llm.bind_tools([tool_schema])
            context = await rag_tool.rag_retrieve_context(
                "Recall our chat on CUDA testing")
            # Inject: prompt += context

        Args:
            query (str):
                The natural language query to search for relevant conversation
                snippets.
                Examples: "our discussion on async Python tools" or "TDD
                strategies from last week".
                Must be non-empty; invalid queries raise ValueError.
            max_chunks (int, optional):
                Number of top chunks to retrieve (defaults to 5).
                Constraints: 1 <= max_chunks <= 20 to avoid excessive context
                length.
                Higher values improve recall but may dilute relevance or exceed
                token limits.

        Returns:
            str:
                A formatted string containing retrieved context via
                CONTEXT_TEMPLATE, ready for LLM prompting. If no context found,
                returns a fallback message acknowledging limitations (e.g., "No
                relevant conversation history found...").
                Length: Typically 200-2000 tokens, depending on top_k and
                chunk size.

        Raises:
            ValueError: If query is empty or invalid (e.g., non-string).
            RuntimeError: If retrieval fails (e.g., vector store connection
                error).
                Includes query and error details for debugging.

        Examples:
            >>> await rag_tool.rag_retrieve_context("Python TDD best practices")
            'Based on the following context from previous conversations, please
            answer the user's question.
            Context: [Snippet 1: "In our last chat, we discussed pytest for TDD..."] [Snippet 2: "..."] 
            Question: Python TDD best practices
            Please provide a comprehensive answer...'

            Edge Case - No Matches:
            >>> await rag_tool.rag_retrieve_context("Unrelated future topic")
            'Based on the following context from previous conversations, please
            answer the user's question.
            Context: No relevant conversation history found for the given query. 
            Question: Unrelated future topic 
            Please provide a comprehensive answer...'

        Integration Notes for Your Repo:
            - Schema for ToolCallProcessor: Use
              RAGRetrieveInput.model_json_schema() for OpenAI-compatible
              binding.
            - Scaling: For production (per Haystack best practices:
              https://haystack.deepset.ai/tutorials),
              add reranking (e.g., via Cohere) post-retrieval.
        """
        if max_chunks is None:
            max_chunks = self.DEFAULT_MAX_CHUNKS

        results_dict = await self.retrieve_context(
            query=query,
            role_filter=self.DEFAULT_ROLE_FILTER,
            max_chunks=max_chunks,
            is_return_matches=self.DEFAULT_IS_RETURN_MATCHES
        )

        return results_dict["message"]

    def create_function_as_tool(
            self,
            role_filter: Optional[str] = None,
            max_chunks: int = 5,
            is_return_matches: bool = False) -> Callable:
        """
        Create a function as a tool that retrieves context from previous conversations.
        """
        return partial(
            self.retrieve_context_sync,
            role_filter=role_filter,
            max_chunks=max_chunks,
            is_return_matches=is_return_matches)