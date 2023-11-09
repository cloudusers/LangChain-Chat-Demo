import sys
import os

from dotenv import load_dotenv, find_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from embeddings import ZhipuAIEmbeddings


def get_embedding(embedding_model_name: str):
    """
    Get embedding model.

    Supported embedding models:

    OPENAI:
        - text-embedding-ada-002
    ZHIPUAI:
        - text_embedding
    QIANFAN:
        - Embedding-V1 (default model)
        - bge-large-en
        - bge-large-zh
    """
    _ = load_dotenv(find_dotenv())
    if embedding_model_name == "text-embedding-ada-002":
        embedding = OpenAIEmbeddings()
    elif embedding_model_name == "text_embedding":
        embedding = ZhipuAIEmbeddings()
    else:
        try:
            embedding = QianfanEmbeddingsEndpoint(model=embedding_model_name)
        except ValueError:
            raise ValueError(f"Unsupported embedding model: {embedding_model_name}")
    return embedding
