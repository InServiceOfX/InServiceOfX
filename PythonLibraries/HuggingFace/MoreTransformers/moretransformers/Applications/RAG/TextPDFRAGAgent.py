from typing import List, Dict, Any, Optional
from .TextPDFRAGProcessor import TextPDFRAGProcessor
from commonapi.Messages import UserMessage, AssistantMessage
from commonapi.Messages.ConversationSystemAndPermanent \
    import ConversationSystemAndPermanent
import asyncio

class TextPDFRAGAgent:
    """AI agent that uses text PDF RAG for answering questions."""

    DEFAULT_SYSTEM_MESSAGE = (
        "You are a helpful AI assistant that answers questions based on PDF documents. "
        "When answering questions, use the context from the PDF documents to provide accurate and relevant information. "
        "If you don't have enough context to answer a question, say so."
    )

    def __init__(
            self,
            pdf_rag_processor: TextPDFRAGProcessor,
            conversation_system_and_permanent: \
                Optional[ConversationSystemAndPermanent]):
        self.pdf_rag_processor = pdf_rag_processor

        if conversation_system_and_permanent is None:
            self.csp = ConversationSystemAndPermanent()
        else:
            self.csp = conversation_system_and_permanent
        
        self.csp.add_system_message(self.DEFAULT_SYSTEM_MESSAGE)
    
    async def answer_question(
            self,
            question: str, 
            max_context_chunks: int = 5) -> str:
        """
        Answer a question using text PDF RAG.
        
        Args:
            question: User's question
            max_context_chunks: Maximum number of context chunks to retrieve
            
        Returns:
            AI-generated answer
        """
        try:
            # Add user question to conversation
            self.csp.append_message(UserMessage(content=question))
            
            # Search for relevant context
            print("Searching for relevant context...")
            relevant_chunks = await self.pdf_rag_processor.search_documents(
                question,
                max_context_chunks
            )
            
            if not relevant_chunks:
                answer = (
                    "I don't have enough context from the PDF documents to "
                    "answer your question. Please try rephrasing or ask about "
                    "something else."
                )
            else:
                context = self._build_context_from_chunks(relevant_chunks)
                enhanced_prompt = self._create_enhanced_prompt(question, context)
                
                # Generate answer (this would integrate with your LLM)
                answer = await self._generate_answer_with_context(enhanced_prompt)
            
            # Add AI answer to conversation
            self.csp.append_message(AssistantMessage(content=answer))
            
            return answer
            
        except Exception as e:
            error_msg = f"Error processing question: {e}"
            print(error_msg)
            return error_msg
    
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = f"Source: {chunk.filename}"
            if chunk.page_number is not None:
                source_info += f" (Page {chunk.page_number})"
            
            context_parts.append(
                f"Context {i}:\n{chunk.content}\n{source_info}\n")
        
        return "\n".join(context_parts)
    
    def _create_enhanced_prompt(self, question: str, context: str) -> str:
        """Create an enhanced prompt with context for the LLM."""
        return f"""Based on the following context from PDF documents, please answer the user's question.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation while providing what information you can from the available context."""
    
    async def _generate_answer_with_context(self, enhanced_prompt: str) -> str:
        """
        Generate answer using the enhanced prompt.
        
        Note: This is a placeholder. You would integrate with your actual LLM here.
        """
        # For now, return a simple response
        # In practice, you would call your LLM (e.g., Groq, OpenAI, local model)
        
        # This is where you'd integrate with your existing LLM infrastructure
        # For example:
        # response = await your_llm_client.chat_completion(enhanced_prompt)
        # return response.choices[0].message.content
        
        return f"Based on the context provided, here's what I found:\n\n{enhanced_prompt}\n\n[Note: This is a placeholder response. Integrate with your actual LLM to generate real answers.]"
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation.get_conversation_as_list_of_dicts()
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation.clear_conversation_history()
    
    async def process_multiple_pdfs(self, pdf_paths: List[str | Path]) -> Dict[str, bool]:
        """Process multiple PDF files and return success status for each."""
        results = {}
        
        for pdf_path in pdf_paths:
            success = await self.pdf_rag_processor.process_pdf_file(pdf_path)
            results[str(pdf_path)] = success
        
        return results