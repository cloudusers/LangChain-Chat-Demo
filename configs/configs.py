PERSIST_DIR = "knowledge_base/chroma"
"""The path to save vector database in your local disk."""
LLMS = ["chatglm_turbo",
        "gpt-3.5-turbo",
        "ERNIE-Bot-turbo"
        ]
"""Supported llms"""
EMBEDDINGS = ["text-embedding-ada-002",
              "text_embedding",
              "Embedding-V1",
              "bge-large-en",
              "bge-large-zh"
              ]
"""Supported embedding models"""
DEFAULT_EMBEDDING = "text-embedding-ada-002"
"""Default embedding model"""
DEFAULT_LLM = "gpt-3.5-turbo"
"""Default llm"""
DEFAULT_TEMPERATURE = 0.1
"""Set the output stability"""
DEFAULT_TOP_K = 3
"""The number of documents to be retrieved from the knowledge base"""
