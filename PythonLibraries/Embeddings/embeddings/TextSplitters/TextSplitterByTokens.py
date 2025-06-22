from typing import List, Optional, Union
from pathlib import Path

from transformers import AutoTokenizer

from .get_token_count import get_token_count
from .get_max_token_limit import get_max_token_limit

class TextSplitterByTokens:
    """
    This splitter ensures each chunk is less than max_tokens while trying to
    keep chunks as large as possible without exceeding the limit.
    Chunks are distinct with no overlap, and can be perfectly reconstructed.

    This is meant to replace TextSplitter and other text splitters by token in
    langchain:
    https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_tokenizer = None,
        max_tokens: Optional[int] = None,
        add_special_tokens: Optional[bool] = None
    ):
        if model_tokenizer is None:
            if model_path is None:
                raise ValueError(
                    "Either model_tokenizer or model_path must be provided")
            model_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only = True,
                )

        self.model_tokenizer = model_tokenizer
        self.add_special_tokens = add_special_tokens
        
        # Set max_tokens if not provided
        if max_tokens is None:
            if model_path is not None:
                max_tokens = get_max_token_limit(model_path)
            else:
                # Try to get from tokenizer config
                try:
                    max_tokens = model_tokenizer.model_max_length
                except AttributeError:
                    # Default fallback
                    max_tokens = 512
        
        self.max_tokens = max_tokens
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks, each with token count <= max_tokens.
            Chunks are distinct and can be perfectly reconstructed.
        """
        if not text or not text.strip():
            return []
        
        # Get total token count
        total_tokens = get_token_count(
            self.model_tokenizer, 
            text, 
            self.add_special_tokens
        )
        
        # If text fits in one chunk, return it
        if total_tokens <= self.max_tokens:
            return [text]
        
        # Split text into chunks
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find the end position for this chunk
            end_pos = self._find_chunk_end(text, current_pos)
            
            # Extract the chunk
            chunk = text[current_pos:end_pos]
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to next position (no overlap)
            current_pos = end_pos
            
            # Stop if we've reached the end
            if current_pos >= len(text):
                break
        
        return chunks
    
    def _find_chunk_end(self, text: str, start_pos: int) -> int:
        """
        Find the end position for a chunk starting at start_pos.
        
        Args:
            text: The full text
            start_pos: Starting position for this chunk
            
        Returns:
            End position for the chunk
        """
        # Start with a reasonable chunk size estimate
        # Rough estimate: assume average of 4 characters per token
        chunk_size = min(len(text) - start_pos, self.max_tokens * 4)
        end_pos = start_pos + chunk_size
        
        # Binary search to find the optimal end position
        left, right = start_pos, min(end_pos, len(text))
        
        while left < right:
            mid = (left + right + 1) // 2
            chunk_text = text[start_pos:mid]
            
            token_count = get_token_count(
                self.model_tokenizer, 
                chunk_text, 
                self.add_special_tokens
            )
            
            if token_count <= self.max_tokens:
                left = mid
            else:
                right = mid - 1
        
        return left
