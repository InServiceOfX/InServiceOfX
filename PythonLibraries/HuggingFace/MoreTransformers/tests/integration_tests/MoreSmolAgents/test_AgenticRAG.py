import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

def create_knowledge_base():
    """
    https://huggingface.co/docs/smolagents/en/examples/rag#step-2-prepare-the-knowledge-base
    """



