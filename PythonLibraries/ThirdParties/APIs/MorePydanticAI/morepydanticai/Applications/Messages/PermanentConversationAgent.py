from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

import asyncpg

class PermanentConversationAgent:
    @dataclass
    class RAGAgentDependencies:
        embedder: SentenceTransformer
        pool: asyncpg.Pool

    def create_rag_agent(self, question: str) -> str:
        return self.agent.run(question)