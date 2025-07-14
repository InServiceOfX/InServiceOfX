from sentence_transformers import SentenceTransformer
import pydantic_core

class MakeMessageEmbeddingsWithSentenceTransformer:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    def make_embedding_from_message(self, message) -> list[float]:
        to_encode = f"{message.role}: {message.content}"
        embedding = self.embedder.encode(to_encode, normalize_embeddings=True)
        # embedding is now a numpy.ndarray, convert to list for JSON
        embedding_list = embedding.tolist()
        embedding_json = pydantic_core.to_json(embedding_list).decode()
        return embedding_json

    def make_embedding_from_message_pair(self, message_0, message_1) \
        -> list[float]:
        to_encode = \
            f"{message_0.role}: {message_0.content}\n{message_1.role}: {message_1.content}"
        embedding = self.embedder.encode(to_encode, normalize_embeddings=True)
        return pydantic_core.to_json(embedding.tolist()).decode()

    def make_embedding_from_content(self, content: str, role: str) \
        -> list[float]:
        to_encode = f"{role}: {content}"
        embedding = self.embedder.encode(to_encode, normalize_embeddings=True)
        return pydantic_core.to_json(embedding.tolist()).decode()

    def make_embedding_from_content_pair(
            self,
            content_0: str,
            content_1: str,
            role_0: str,
            role_1: str) \
        -> list[float]:
        to_encode = \
            f"{role_0}: {content_0}\n{role_1}: {content_1}"
        embedding = self.embedder.encode(to_encode, normalize_embeddings=True)
        return pydantic_core.to_json(embedding.tolist()).decode()