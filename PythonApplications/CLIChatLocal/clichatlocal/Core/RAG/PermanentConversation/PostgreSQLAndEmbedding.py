from clichatlocal.Core import ProcessConfigurations
from embeddings.TextSplitters import TextSplitterByTokens
from sentence_transformers import SentenceTransformer
from tools.RAG.PermanentConversation.EmbedPermanentConversation \
    import EmbedPermanentConversation
from tools.RAG.PermanentConversation import (
    PostgreSQLInterface,
    RAGProcessor,
    RAGTool
)

class PostgreSQLAndEmbedding:
    def __init__(
        self,
        input_postgres_connection,
        process_configurations: ProcessConfigurations):
        """
        Args:
            process_configurations: This should be the variable name of the
            single ProcessConfigurations object, instance, that we'll
            exclusively use for this application.
        """
        self._pgsql_interface = PostgreSQLInterface(input_postgres_connection)
        self._text_splitter = None
        self._embedding_model = None
        # EmbedPermanentConversation
        self._embed_pc = None

        self._embedding_models_configuration = \
            process_configurations.configurations[
                "embedding_models_configuration"]
        self._device_map = process_configurations.configurations[
            "from_pretrained_model_configuration"].device_map

        self._prompt_mode = "Direct"

        self._rag_processor = None
        self._rag_tool = None

    def get_prompt_mode(self) -> str:
        return self._prompt_mode

    def _set_prompt_mode(self, prompt_mode: str):
        self._prompt_mode = prompt_mode

    def _toggle_prompt_mode(self):
        self._prompt_mode = "Direct" if self._prompt_mode == "RAG" else "RAG"

    async def create_tables(self):
        await self._pgsql_interface.create_tables()

    async def get_latest_message_chunks(self, limit: int):
        return await self._pgsql_interface.get_latest_message_chunks(limit)

    def setup_embedding_model(self):
        model_path = self._embedding_models_configuration.text_embedding_model
        self._text_splitter = TextSplitterByTokens(model_path=model_path)
        if self._device_map is not None:
            self._embedding_model = SentenceTransformer(
                str(model_path),
                device = str(self._device_map))
        else:
            self._embedding_model = SentenceTransformer(str(model_path))

    def create_EmbedPermanentConversation(self, permanent_conversation):
        self._embed_pc = EmbedPermanentConversation(
            self._text_splitter,
            self._embedding_model,
            permanent_conversation
        )

    async def embed_conversation(self):
        if self._embed_pc is None:
            return None
        message_chunks, message_pair_chunks = \
            self._embed_pc.embed_conversation()
        for message_chunk in message_chunks:
            await self._pgsql_interface.insert_message_chunk(message_chunk)
        for message_pair_chunk in message_pair_chunks:
            await self._pgsql_interface.insert_message_pair_chunk(
                message_pair_chunk)
        return message_chunks, message_pair_chunks

    def _create_RAG_Processor(self):
        if self._pgsql_interface is not None and self._embed_pc is not None:
            self._rag_processor = RAGProcessor(
                self._pgsql_interface,
                self._embed_pc)
            return True
        else:
            return None

    def _create_RAG_Tool(self):
        if self._rag_processor is not None:
            self._rag_tool = RAGTool(self._rag_processor)
            return True
        else:
            return None

    def create_RAG(self):
        create_RAG_processor_result = self._create_RAG_Processor()
        if create_RAG_processor_result is None:
            return None
        return self._create_RAG_Tool()