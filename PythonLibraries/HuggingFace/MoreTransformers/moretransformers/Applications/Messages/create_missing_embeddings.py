from commonapi.Messages import ConversationSystemAndPermanent
from commonapi.Messages.PermanentConversation import PermanentConversation
from .MakeMessageEmbeddingsWithSentenceTransformer import \
    MakeMessageEmbeddingsWithSentenceTransformer

def create_missing_embeddings(
        conversation: ConversationSystemAndPermanent | PermanentConversation,
        embeddings_maker: MakeMessageEmbeddingsWithSentenceTransformer):

    if isinstance(conversation, ConversationSystemAndPermanent):
        messages = conversation.permanent_conversation.messages
        message_pairs = conversation.permanent_conversation.message_pairs
    elif isinstance(conversation, PermanentConversation):
        messages = conversation.messages
        message_pairs = conversation.message_pairs
    else:
        raise ValueError("Invalid conversation type")

    for message in messages:
        if message.embedding is None:
            message.embedding = \
                embeddings_maker.make_embedding_from_content(
                    content=message.content,
                    role=message.role)

    for message_pair in message_pairs:
        if message_pair.embedding is None:
            message_pair.embedding = \
                embeddings_maker.make_embedding_from_content_pair(
                    content_0=message_pair.content_0,
                    content_1=message_pair.content_1,
                    role_0=message_pair.role_0,
                    role_1=message_pair.role_1)