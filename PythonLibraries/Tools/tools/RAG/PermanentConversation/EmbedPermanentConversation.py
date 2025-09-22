from dataclasses import dataclass
from commonapi.Messages.PermanentConversation import (
    ConversationMessage,
    PermanentConversation)
from embeddings.TextSplitters import TextSplitterByTokens
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Tuple, Dict
import hashlib
import time
from typing import List
from collections import defaultdict

@dataclass
class ConversationMessageChunk:
    """Represents a chunk of a longer message for embedding purposes."""
    conversation_id: int
    # 0-based index within the original message
    chunk_index: int
    total_chunks: int
    parent_message_hash: str
    content: str
    datetime: float
    hash: str
    role: str
    chunk_type: str
    embedding: Optional[list[float]] = None

class EmbedPermanentConversation:
    def __init__(
            self,
            text_splitter: TextSplitterByTokens,
            embedding_model: SentenceTransformer,
            permanent_conversation: PermanentConversation):
        self._text_splitter = text_splitter
        self._embedding_model = embedding_model
        self._pc = permanent_conversation

    def _should_split_message(self, content: str) -> bool:
        return self._text_splitter.get_token_count(content) > \
            self._text_splitter.max_tokens

    def _create_chunk_hash(
            self,
            content: str,
            parent_hash: str,
            chunk_index: int) -> str:
        """Create a unique hash for a chunk."""
        combined_content = \
            f"{parent_hash}_{chunk_index}_{content}_{time.time()}"
        return hashlib.sha256(combined_content.encode()).hexdigest()

    def embed_conversation(self) \
        -> Tuple[
            List[ConversationMessageChunk],
            List[ConversationMessageChunk]]:
        """Process all messages and create embeddings with chunking if
        needed."""
        message_chunks = []
        message_pair_chunks = []

        # Process individual messages
        for message in self._pc.messages:
            if self._should_split_message(message.content):
                chunks = self._text_splitter.split_text(message.content)
                for chunk_index, chunk_content in enumerate(chunks):
                    chunk = ConversationMessageChunk(
                        conversation_id=message.conversation_id,
                        chunk_index=chunk_index,
                        total_chunks=len(chunks),
                        parent_message_hash=message.hash,
                        content=chunk_content,
                        datetime=message.datetime,
                        hash=self._create_chunk_hash(
                            chunk_content,
                            message.hash,
                            chunk_index),
                        role=message.role,
                        chunk_type="message"
                    )
                    # Generate embedding for this chunk
                    chunk.embedding = self._generate_embedding(
                        chunk_content,
                        message.role)
                    message_chunks.append(chunk)
            else:
                chunk = ConversationMessageChunk(
                    conversation_id=message.conversation_id,
                    chunk_index=0,
                    total_chunks=1,
                    parent_message_hash=message.hash,
                    content=message.content,
                    datetime=message.datetime,
                    hash=message.hash,
                    role=message.role,
                    chunk_type="message"
                )
                chunk.embedding = self._generate_embedding(
                    message.content,
                    message.role)
                message_chunks.append(chunk)

        # Process message pairs
        for message_pair in self._pc.message_pairs:
            combined_content = \
                f"{message_pair.role_0}: {message_pair.content_0}\n{message_pair.role_1}: {message_pair.content_1}"
            if self._should_split_message(combined_content):
                chunks = self._text_splitter.split_text(combined_content)
                for chunk_index, chunk_content in enumerate(chunks):
                    chunk = ConversationMessageChunk(
                        conversation_id=message_pair.conversation_pair_id,
                        chunk_index=chunk_index,
                        total_chunks=len(chunks),
                        parent_message_hash=message_pair.hash,
                        content=chunk_content,
                        datetime=message_pair.datetime,
                        hash=self._create_chunk_hash(
                            chunk_content,
                            message_pair.hash,
                            chunk_index),
                        role=f"{message_pair.role_0}_{message_pair.role_1}",
                        chunk_type="message_pair"
                    )
                    chunk.embedding = self._generate_embedding_for_message_pair(
                        chunk_content,)
                    message_pair_chunks.append(chunk)
            else:
                chunk = ConversationMessageChunk(
                    conversation_id=message_pair.conversation_pair_id,
                    chunk_index=0,
                    total_chunks=1,
                    parent_message_hash=message_pair.hash,
                    content=combined_content,
                    datetime=message_pair.datetime,
                    hash=message_pair.hash,
                    role=f"{message_pair.role_0}_{message_pair.role_1}",
                    chunk_type="message_pair"
                )
                chunk.embedding = self._generate_embedding_for_message_pair(
                    combined_content,)
                message_pair_chunks.append(chunk)

        return message_chunks, message_pair_chunks

    def _generate_embedding(self, content: str, role: str) -> List[float]:
        """Generate embedding for content with role context."""
        # Format content with role for better context
        formatted_content = f"{role}: {content}"
        embedding = self._embedding_model.encode(
            formatted_content,
            normalize_embeddings=True)
        return embedding.tolist()

    def _generate_embedding_for_message_pair(
            self,
            content: str,
            ) -> List[float]:
        """Generate embedding for message pair."""
        embedding = self._embedding_model.encode(
            content,
            normalize_embeddings=True)
        return embedding.tolist()

    @staticmethod
    def recreate_conversation_messages_from_chunks(
        message_chunks: List[ConversationMessageChunk]
    ) -> List[ConversationMessage]:
        """
        Use this function for testing purposes.
        """
        # Filter only message chunks (not message_pair chunks)
        message_only_chunks = [
            chunk for chunk in message_chunks 
            if chunk.chunk_type == "message"
        ]

        # Group chunks by a "composite key":
        # (parent_message_hash, conversation_id, datetime)
        chunks_by_composite_key = defaultdict(list)
        for chunk in message_only_chunks:
            composite_key = \
                (
                    chunk.parent_message_hash,
                    chunk.conversation_id,
                    chunk.datetime)
            chunks_by_composite_key[composite_key].append(chunk)
        
        # Sort chunks within each group by chunk_index
        for composite_key in chunks_by_composite_key:
            chunks_by_composite_key[composite_key].sort(
                key=lambda x: x.chunk_index)
        
        # Reconstruct ConversationMessage instances
        reconstructed_messages = []
        
        for composite_key, chunks in chunks_by_composite_key.items():
            parent_hash, conversation_id, datetime_val = composite_key

            # Sort by chunk_index to ensure correct order
            chunks.sort(key=lambda x: x.chunk_index)

            # Get the first chunk to extract metadata
            first_chunk = chunks[0]
            
            # Reconstruct the original content by concatenating chunks
            reconstructed_content = "".join(chunk.content for chunk in chunks)
            
            # Create ConversationMessage instance
            message = ConversationMessage(
                conversation_id=conversation_id,
                content=reconstructed_content,
                datetime=datetime_val,
                hash=parent_hash,
                role=first_chunk.role
            )
            
            reconstructed_messages.append(message)
        
        # Sort by conversation_id to maintain original order
        reconstructed_messages.sort(key=lambda x: x.conversation_id)
        
        return reconstructed_messages

    @staticmethod
    def recreate_conversation_messages_pairs_from_chunks(
        message_pair_chunks: List[ConversationMessageChunk]):
        """
        Use this function for testing purposes.
        """
        # Filter only message_pair chunks
        message_pair_chunks_filtered = [
            chunk for chunk in message_pair_chunks 
            if chunk.chunk_type == "message_pair"
        ]
        
        # Group chunks by a "composite key"
        chunks_by_composite_key = defaultdict(list)
        for chunk in message_pair_chunks_filtered:
            composite_key = \
                (
                    chunk.parent_message_hash,
                    chunk.conversation_id,
                    chunk.datetime)
            chunks_by_composite_key[composite_key].append(chunk)

        # Sort chunks within each group by chunk_index
        for composite_key in chunks_by_composite_key:
            chunks_by_composite_key[composite_key].sort(
                key=lambda x: x.chunk_index)
        
        reconstructed_message_pairs = []
        
        for composite_key, chunks in chunks_by_composite_key.items():
            parent_hash, conversation_id, datetime_val = composite_key
            chunks.sort(key=lambda x: x.chunk_index)
            reconstructed_content = "".join(chunk.content for chunk in chunks)
            reconstructed_message_pairs.append(
                {
                    "conversation_pair_id": conversation_id,
                    "content": reconstructed_content,
                    "datetime": datetime_val,
                    "hash": parent_hash,
                    "role": chunks[0].role,
                }
            )

        return reconstructed_message_pairs


