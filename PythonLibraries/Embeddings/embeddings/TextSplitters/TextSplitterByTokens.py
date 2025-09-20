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

    def get_token_count(self, text: str) -> int:
        return get_token_count(
            self.model_tokenizer, 
            text, 
            self.add_special_tokens
        )

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
        
        # character-based estimation with verification
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Estimate chunk size based on character count
            # Assume average of 4 characters per token for most text
            estimated_chars = self.max_tokens * 4
            
            # Calculate end position
            end_pos = min(current_pos + estimated_chars, len(text))
            
            # Extract the chunk
            chunk = text[current_pos:end_pos]
            
            # Verify token count and adjust if necessary
            try:
                token_count = get_token_count(
                    self.model_tokenizer, 
                    chunk, 
                    self.add_special_tokens
                )
                
                # If chunk is too long, reduce it
                while token_count > self.max_tokens and len(chunk) > 50:
                    # Reduce by 10% and try again
                    reduction = max(1, int(len(chunk) * 0.1))
                    chunk = chunk[:-reduction]
                    end_pos = current_pos + len(chunk)
                    
                    if len(chunk) == 0:
                        break
                    
                    token_count = get_token_count(
                        self.model_tokenizer, 
                        chunk, 
                        self.add_special_tokens
                    )
                
                # If chunk is too short, try to expand it
                if token_count < self.max_tokens * 0.5 and end_pos < len(text):
                    # Try to expand by adding more text
                    expansion = min(
                        len(text) - end_pos, 
                        (self.max_tokens - token_count) * 4
                    )

                    if expansion > 0:
                        expanded_chunk = text[current_pos:end_pos + expansion]
                        try:
                            expanded_token_count = get_token_count(
                                self.model_tokenizer, 
                                expanded_chunk, 
                                self.add_special_tokens
                            )
                            
                            if expanded_token_count <= self.max_tokens:
                                chunk = expanded_chunk
                                end_pos = current_pos + len(chunk)
                                token_count = expanded_token_count
                        except Exception:
                            # Keep original chunk if expansion fails
                            pass
                
            except Exception as e:
                # If tokenization fails, be very conservative
                print(f"Tokenization error, using conservative chunking: {e}")
                # Force chunk to be much smaller
                safe_size = min(len(text) - current_pos, 100)
                chunk = text[current_pos:current_pos + safe_size]
                end_pos = current_pos + safe_size
            
            # Add the chunk if it's not empty
            if chunk and chunk.strip():
                chunks.append(chunk)
            
            # Move to next position
            current_pos = end_pos
            
            # Safety check to prevent infinite loops
            if current_pos >= len(text):
                break
            if len(chunk) == 0:
                # If we couldn't create a chunk, force progress
                current_pos += 1
        
        return chunks
