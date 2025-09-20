from dataclasses import dataclass
from commonapi.Messages.PermanentConversation import PermanentConversation
from embeddings.TextSplitters import TextSplitterByTokens
from typing import Optional

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
    embedding: Optional[list[float]] = None

class EmbedPermanentConversation:
    def __init__(
            self,
            text_splitter: TextSplitterByTokens,
            permanent_conversation: PermanentConversation):
        self._text_splitter = text_splitter
        self._pc = permanent_conversation

    def _should_split_message(self, content: str) -> bool:
        return self._text_splitter.get_token_count(content) > \
            self._text_splitter.max_tokens

    # def embed_conversation(self, conversation: list[ConversationMessageChunk]):
    #     for message in conversation:
    #         chunks = self.text_splitter.split_text(message.content)
    #         for chunk_index, chunk in enumerate(chunks):
    #             chunk_message = ConversationMessageChunk(
    #                 conversation_id=message.conversation_id,