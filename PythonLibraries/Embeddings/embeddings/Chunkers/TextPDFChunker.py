from embeddings.TextSplitters import TextSplitterByTokens

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib

@dataclass
class TextPDFChunk:
    """Represents a chunk of text from a PDF document."""
    content: str
    chunk_id: int
    source_file: str
    page_number: Optional[int] = None
    chunk_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.chunk_hash is None:
            self.chunk_hash = self._create_hash()
    
    def _create_hash(self) -> str:
        """Create a hash from the chunk content and metadata."""
        content_str = f"{self.content}_{self.source_file}_{self.chunk_id}"
        return hashlib.sha256(content_str.encode()).hexdigest()

class TextPDFChunker:
    """Chunk PDF documents into smaller pieces for embedding."""    
    def __init__(self, model_path: str | Path, max_tokens: int = 512):
        """
        Args:
            model_path: Path to the tokenizer model
            e.g. a tokenizer model you could use could be
            "BAAI/bge-large-en-v1.5"
            max_tokens: Maximum tokens per chunk.
        """
        self.text_splitter = TextSplitterByTokens(
            model_path=model_path,
            max_tokens=max_tokens
        )
    
    def chunk_pdf_text(
        self,
        pdf_text: str,
        source_file: str,
        page_number: Optional[int] = None) -> List[TextPDFChunk]:
        """
        Split PDF text into chunks.

        Args:
            pdf_text: Text content from PDF
            source_file: Name/path of source PDF file
            page_number: Page number if known
            
        Returns:
            List of PDFChunk objects
        """
        if not pdf_text or not pdf_text.strip():
            return []
        
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(pdf_text)
        
        # Convert to PDFChunk objects
        pdf_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = TextPDFChunk(
                content=chunk_text,
                chunk_id=i,
                source_file=source_file,
                page_number=page_number,
                metadata={
                    'chunk_size_tokens': len(
                        self.text_splitter.model_tokenizer.tokenize(
                            chunk_text)),
                    'total_chunks': len(text_chunks),
                    'chunk_index': i
                }
            )
            pdf_chunks.append(chunk)
        
        return pdf_chunks
    
    def chunk_pdf_by_pages(self, page_texts: List[str], source_file: str) \
        -> List[TextPDFChunk]:
        """
        Chunk PDF text page by page.

        Args:
            page_texts: List of page texts
            source_file: Name/path of source PDF file

        Returns:
            List of TextPDFChunk objects
        """
        all_chunks = []
        chunk_id_counter = 0
        
        for page_num, page_text in enumerate(page_texts):
            if not page_text or not page_text.strip():
                continue
            
            page_chunks = self.chunk_pdf_text(page_text, source_file, page_num)
            
            # Update chunk IDs to be globally unique
            for chunk in page_chunks:
                chunk.chunk_id = chunk_id_counter
                chunk_id_counter += 1
                all_chunks.append(chunk)
        
        return all_chunks