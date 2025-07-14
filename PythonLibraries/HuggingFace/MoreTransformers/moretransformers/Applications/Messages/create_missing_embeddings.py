from commonapi.Messages import ConversationSystemAndPermanent
from .MakeMessageEmbeddingsWithSentenceTransformer import \
    MakeMessageEmbeddingsWithSentenceTransformer

def create_missing_embeddings(
        csp: ConversationSystemAndPermanent,
        embeddings_maker: MakeMessageEmbeddingsWithSentenceTransformer):
    for message in csp.permanent_conversation.messages:
        if message.embedding is None:
            message.embedding = \
                embeddings_maker.make_embedding_from_content(
                    content=message.content,
                    role=message.role)

    for message_pair in csp.permanent_conversation.message_pairs:
        if message_pair.embedding is None:
            message_pair.embedding = \
                embeddings_maker.make_embedding_from_content_pair(
                    content_0=message_pair.content_0,
                    content_1=message_pair.content_1,
                    role_0=message_pair.role_0,
                    role_1=message_pair.role_1)