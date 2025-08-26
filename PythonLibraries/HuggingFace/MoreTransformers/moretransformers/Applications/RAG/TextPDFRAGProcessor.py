from pathlib import Path
from typing import List, Optional, Dict, Any


from commonapi.Databases.RAG.TextPDF import PostgreSQLInterface
from corecode.Parsers.pdf.DocumentTextExtraction import DocumentTextExtraction
from embeddings.Chunkers import TextPDFChunker
from sentence_transformers import SentenceTransformer
from warnings import warn
import asyncio

class TextPDFRAGProcessor:
    """Main processor for PDF RAG operations."""
    
    def __init__(
            self, 
            embedding_model_path: str | Path,
            postgres_connection,
            embedder: SentenceTransformer,
            max_tokens_per_chunk: int = 512):
        """
        Args:
            embedding_model_path: Path to SentenceTransformer model
            postgres_connection: PostgreSQL connection
            max_tokens_per_chunk: Maximum tokens per chunk
        """
        self.chunker = TextPDFChunker(
            embedding_model_path,
            max_tokens_per_chunk)
        self.database = PostgreSQLInterface(postgres_connection)
        self.embedder = embedder

    async def process_pdf_file(self, pdf_path: str | Path) -> bool:
        """
        Process a PDF file: extract text, chunk it, create embeddings, and store
        in database.

        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if successful, False otherwise
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            warn(f"PDF file does not exist: {pdf_path}")
            return False
        
        try:
            print(f"Processing PDF: {pdf_path.name}")
            
            # Extract text
            pdf_pages = DocumentTextExtraction.extract_text_by_pages(pdf_path)
            if not pdf_pages:
                warn("Failed to extract text from PDF")
                return False
            
            # Chunk text
            chunks = self.chunker.chunk_pdf_by_pages(pdf_pages, pdf_path.name)
            
            # Create embeddings and store in database
            print("Creating embeddings and storing in database...")
            success = await self._store_chunks_with_embeddings(chunks, pdf_path)

            if success:
                print(f"Successfully processed PDF: {pdf_path.name}")
            else:
                warn(f"Failed to store chunks for PDF: {pdf_path.name}")
            
            return success
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            return False
    
    async def _store_chunks_with_embeddings(
        self,
        chunks: list,
        pdf_path: Path) -> bool:
        """Store chunks with their embeddings in the database."""
        try:
            # Insert document record
            document_id = await self.database.insert_text_pdf(
                filename=pdf_path.name,
                total_chunks=len(chunks)
            )
            
            if document_id is None:
                warn(f"Failed to insert document record for {pdf_path.name}")
                return False
            
            # Process chunks with embeddings
            for chunk in chunks:
                # Create embedding
                embedding = self.embedder.encode(
                    chunk.content,
                    normalize_embeddings=True
                )
                
                # Store in database
                await self.database.insert_text_pdf_chunk(
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    chunk_hash=chunk.chunk_hash,
                    embedding=embedding.tolist(),
                    metadata=chunk.metadata
                )
            
            return True
            
        except Exception as e:
            warn(f"Error storing chunks with embeddings: {e}")
            return False
    
    async def search_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks based on a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Create embedding for the query
            query_embedding = self.embedder.encode(
                query,
                normalize_embeddings=True
            )
            
            # Search for similar chunks
            results = await self.database.search_similar_chunks(
                query_embedding.tolist(), limit
            )
            
            return results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    async def get_document_summary(self, filename: str) -> Dict[str, Any]:
        """Get summary information about a processed document."""
        try:
            chunks = await self.database.get_document_chunks(filename)

            if not chunks:
                return {}
            
            # Calculate summary statistics
            total_chunks = len(chunks)
            total_content_length = sum(len(chunk['content']) \
                for chunk in chunks)
            avg_chunk_length = total_content_length / total_chunks \
                if total_chunks > 0 else 0

            # Get page range
            page_numbers = [
                chunk['page_number'] \
                    for chunk in chunks if chunk['page_number'] is not None]
            page_range = \
                f"{min(page_numbers)}-{max(page_numbers)}" \
                if page_numbers else "Unknown"
            
            return {
                'filename': filename,
                'total_chunks': total_chunks,
                'total_content_length': total_content_length,
                'average_chunk_length': avg_chunk_length,
                'page_range': page_range,
                'chunks': chunks
            }
            
        except Exception as e:
            print(f"Error getting document summary: {e}")
            return {}