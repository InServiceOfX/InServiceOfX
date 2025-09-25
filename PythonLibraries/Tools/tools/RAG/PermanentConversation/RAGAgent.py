from typing import Optional, Any, List, Dict
from commonapi.Messages import UserMessage, AssistantMessage
from commonapi.Messages import ConversationAndSystemMessages

from .RAGProcessor import RAGProcessor

from textwrap import dedent

class RAGAgent:
    """AI agent that uses permanent conversation RAG for answering questions."""
    
    DEFAULT_SYSTEM_MESSAGE = dedent("""
        You are a helpful AI assistant that answers questions based on previous
        conversation history. When answering questions, use the context from
        previous conversations to provide accurate and relevant information.
    """)
    
    def __init__(
        self,
        rag_processor: RAGProcessor,
        conversation_and_system_messages: Optional[ConversationAndSystemMessages] = None
    ):
        self.rag_processor = rag_processor
        
        if conversation_and_system_messages is None:
            self.csm = ConversationAndSystemMessages()
        else:
            self.csm = conversation_and_system_messages
        
        self.csm.add_system_message(self.DEFAULT_SYSTEM_MESSAGE)
    
    async def answer_question(
        self,
        question: str,
        role_filter: Optional[str] = None,
        max_context_chunks: int = 5
    ) -> str:
        """
        Answer a question using permanent conversation RAG.
        
        Args:
            question: User's question
            role_filter: Optional role filter ('user', 'assistant', 'system')
            max_context_chunks: Maximum number of context chunks to retrieve
            
        Returns:
            AI-generated answer
        """
        try:
            # Add user question to conversation
            self.csp.append_message(UserMessage(content=question))
            
            # Retrieve relevant context using your RAGProcessor
            print("Searching for relevant conversation context...")
            context = await self.rag_processor.process_query_to_context(
                query=question,
                role_filter=role_filter,
                limit=max_context_chunks
            )
            
            if not context or len(context.strip()) < 50:
                answer = (
                    "I don't have enough context from previous conversations to "
                    "answer your question. Please try rephrasing or ask about "
                    "something else."
                )
            else:
                # Create enhanced prompt with context
                enhanced_prompt = self._create_enhanced_prompt(question, context)
                
                # Generate answer using LLM
                answer = await self._generate_answer_with_context(enhanced_prompt)
            
            # Add AI answer to conversation
            self.csp.append_message(AssistantMessage(content=answer))
            
            return answer
            
        except Exception as e:
            error_msg = f"Error processing question: {e}"
            print(error_msg)
            return error_msg
    
    def _create_enhanced_prompt(self, question: str, context: str) -> str:
        """Create an enhanced prompt with context for the LLM."""
        return f"""Based on the following context from previous conversations, please answer the user's question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation while providing what information you can from the available context."""
    
    async def _generate_answer_with_context(self, enhanced_prompt: str) -> str:
        """Generate answer using the enhanced prompt."""
        # This is where you'd integrate with your actual LLM
        # For example:
        # response = await self.llm_engine.generate(enhanced_prompt)
        # return response
        
        # Placeholder for now
        return f"Based on the context provided, here's what I found:\n\n{enhanced_prompt}\n\n[Note: This is a placeholder response. Integrate with your actual LLM to generate real answers.]"
    
    def get_conversation_as_list_of_dicts(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.csp.get_conversation_as_list_of_dicts()
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.csp.clear_conversation_history()
