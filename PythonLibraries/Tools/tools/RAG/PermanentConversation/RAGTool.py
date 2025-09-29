from typing import Optional, Any, List, Dict

from .RAGProcessor import RAGProcessor

from string import Template
from textwrap import dedent

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

    def __init__(self, rag_processor: RAGProcessor):
        self.rag_processor = rag_processor

    async def retrieve_context(
        self,
        query: str,
        role_filter: Optional[str] = None,
        max_chunks: int = 5,
        is_return_matches: bool = False,
    ) -> str:
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
