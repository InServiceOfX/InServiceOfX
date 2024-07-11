from langchain.chains import RetrievalQA
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI

import os

class AtlasVectorSearch:
    """
    @url https://www.mongodb.com/developer/products/atlas/rag-atlas-vector-search-langchain-openai/
    @ref RAG with Atlas Vector Search, LangChain, and OpenAI. Harshad Dhavale
    """
    def __init__(
        self,
        collection,
        embedding_model=None,
        openai_api_key=None
        ):
        if openai_api_key == None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")

        if embedding_model == None:
            self._embedding_model = OpenAIEmbeddings(
                openai_api_key=openai_api_key)
        else:
            self._embedding_model = embedding_model

        self.atlas_vector_store = MongoDBAtlasVectorSearch(
            collection,
            self._embedding_model)

        # Define the LLM that we want to use -- note that this is the Language Generation Model and NOT an Embedding Model
        # If it's not specified (for example like in the code below),
        # then the default OpenAI model used in LangChain is OpenAI GPT-3.5-turbo, as of August 30, 2023
        # temperature=0 means more definitive, less creative.
        self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0)

        # Leveraging Atlas Vector Search paired with Langchain's QARetriever

        # Get VectorStoreRetriever: Specifically, Retriever for MongoDB VectorStore.
        # Implements _get_relevant_documents which retrieves documents relevant to a query.
        self.retriever = self.atlas_vector_store.as_retriever()

        # Load "stuff" documents chain. Stuff documents chain takes a list of documents,
        # inserts them all into a prompt and passes that prompt to an LLM.

        self.qa = RetrievalQA.from_chain_type(
            self.llm,
            chain_type="stuff",
            retriever=self.retriever)

        # Stay safe, delete the API key after use/consumption.
        del openai_api_key

    def query_data(self, query):
        # Convert question to vector using OpenAI embeddings
        # Perform Atlas Vector Search using Langchain's vectorStore
        # similarity_search returns MongoDB documents most similar to the query 

        docs = self.atlas_vector_store.similarity_search(query, K=1)

        if (len(docs) > 0):
            as_output = docs[0].page_content
        else:
            as_output = str(docs)

        # Execute the chain
        retriever_output = self.qa.run(query)

        return as_output, retriever_output