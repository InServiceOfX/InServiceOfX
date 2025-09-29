from typing import Optional

class RAGProcessor:
    DEFAULT_LIMIT = 5

    def __init__(
        self,
        postgresql_interface,
        embed_permanent_conversation,
        max_content_length: Optional[int] = None):
        self._pgsql_interface = postgresql_interface
        self._embed_pc = embed_permanent_conversation

        if max_content_length is None:
            # I had tried 4 * 32768, but make it longer.
            # 4 * 131072 is quite long and sufficient.
            self._max_content_length = 4 * 65536
        else:
            self._max_content_length = max_content_length

    async def _retrieve_relevant_message_chunks(
            self,
            query: str,
            role_filter: Optional[str] = None,
            limit: int = DEFAULT_LIMIT):
        # These embeddings of the chunks as query is split into chunks.
        query_embeddings = self._embed_pc.make_query_embeddings(query)

        search_results = []
        for embedding in query_embeddings:
            search_results.append(
                await self._pgsql_interface.vector_similarity_search_message_chunks(
                    query_embedding=embedding,
                    role_filter=role_filter,
                    limit=limit))

        # Each search result corresponds to a chunk of the original query. Each
        # search result is a list of similarity matches.
        return search_results

    def _build_context_from_message_chunks(
            self,
            search_results,
            role_filter: Optional[str] = None):
        """
        Args:
            search_results: Typically the result of
            _retrieve_relevant_message_chunks.
        """
        context_parts = []
        current_length = 0

        for chunk_results in search_results:
            for index, similarity_match in enumerate(chunk_results):
                if index != 0 and role_filter != None:
                    match_content = similarity_match["content"]
                else:
                    match_content = \
                        f"[{similarity_match['role']}] {similarity_match['content']}"

                context_parts.append(match_content)
                current_length += len(match_content)
                if current_length > self._max_content_length:
                    return "\n".join(context_parts)

        return "\n".join(context_parts)

    async def process_query_to_context(
            self,
            query: str,
            role_filter: Optional[str] = None,
            limit: int = DEFAULT_LIMIT,
            is_return_matches: bool = False):
        search_results = \
            await self._retrieve_relevant_message_chunks(
                query,
                role_filter,
                limit)
        context = self._build_context_from_message_chunks(
            search_results,
            role_filter)
        return context if not is_return_matches else (context, search_results)